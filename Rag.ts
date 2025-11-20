

import { config } from "dotenv";
import fs from "fs";
import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";

// LangChain imports
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain as createLCRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

config();

// ======================================================
// CONFIG
// ======================================================
const DATA_DIR = "data";

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "rishabh-portfolio";
const HF_API_KEY = process.env.HF_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const EMBEDDING_DIMENSION = 384;

// ======================================================
// GLOBALS
// ======================================================
let embeddings: HuggingFaceInferenceEmbeddings;
let pineconeClient: PineconeClient;
let pineconeIndex: any;

// Track processed files (so we don’t re-embed the same file)
const processedFilesPath = "./processed_files.json";
let processedFiles = new Set<string>();

if (fs.existsSync(processedFilesPath)) {
  try {
    processedFiles = new Set(
      JSON.parse(fs.readFileSync(processedFilesPath, "utf-8"))
    );
  } catch (err) {
    console.error("Failed to load processed_files.json:", err);
  }
}

function saveProcessedFiles() {
  fs.writeFileSync(
    processedFilesPath,
    JSON.stringify([...processedFiles], null, 2),
    "utf-8"
  );
}

// ======================================================
// 1. LOAD DOCUMENTS
// ======================================================

async function loadTextFiles(dir: string): Promise<Document[]> {
  const docs: Document[] = [];
  const items = fs.readdirSync(dir);

  for (let item of items) {
    const full = path.join(dir, item);
    const stat = fs.statSync(full);

    if (stat.isDirectory()) {
      docs.push(...await loadTextFiles(full));
      continue;
    }

    if (/\.(txt|md|mdx)$/i.test(item)) {
      const content = fs.readFileSync(full, "utf8");
      docs.push(new Document({ pageContent: content, metadata: { source: full } }));
    }
  }

  return docs;
}

async function loadPDFs(dir: string): Promise<Document[]> {
  const docs: Document[] = [];
  const items = fs.readdirSync(dir);

  for (let item of items) {
    const full = path.join(dir, item);
    const stat = fs.statSync(full);

    if (stat.isDirectory()) {
      docs.push(...await loadPDFs(full));
      continue;
    }

    if (item.toLowerCase().endsWith(".pdf")) {
      const buffer = fs.readFileSync(full);
      const pdfParse = (await import("pdf-parse")).default;
      const data = await pdfParse(buffer);
      docs.push(new Document({ pageContent: data.text, metadata: { source: full } }));
    }
  }

  return docs;
}

async function ingestDocuments() {
  console.log("📥 Loading initial documents...");
  const textDocs = await loadTextFiles(DATA_DIR);
  const pdfDocs = await loadPDFs(DATA_DIR);
  return [...textDocs, ...pdfDocs];
}

// ======================================================
// 2. CHUNKING
// ======================================================

async function chunkDocuments(docs: Document[]): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 100,
  });

  const chunks = await splitter.splitDocuments(docs);
  chunks.forEach((c, i) => (c.metadata.chunkIndex = i));
  return chunks;
}

// ======================================================
// 3. PINECONE INIT
// ======================================================

async function initializePinecone() {
  pineconeClient = new PineconeClient(); // auto uses env
  const indexes = await pineconeClient.listIndexes();

  const indexNames = indexes.indexes?.map((i: any) => i.name) ?? [];


  if (!indexNames.includes(PINECONE_INDEX_NAME)) {
    console.log("Creating Pinecone index...");
    await pineconeClient.createIndex({
      name: PINECONE_INDEX_NAME,
      dimension: EMBEDDING_DIMENSION,
      metric: "cosine",
      spec: {
        serverless: {
          cloud: "aws",
          region: "us-east-1",
        },
      },
    });

    console.log("Waiting 45 seconds for Pinecone index to initialize...");
    await new Promise((resolve) => setTimeout(resolve, 45000));
  }

  pineconeIndex = pineconeClient.index(PINECONE_INDEX_NAME);
  console.log("✔ Connected to Pinecone index");
}

// ======================================================
// 4. UPSERT
// ======================================================

async function upsertToPinecone(chunks: Document[]) {
  console.log(`📤 Upserting ${chunks.length} chunks...`);

  const batchSize = 100;

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);

    const texts = batch.map((d) => d.pageContent);
    const embedded = await embeddings.embedDocuments(texts);

    const vectors = batch.map((chunk, idx) => ({
      id: `chunk-${Date.now()}-${i + idx}`,
      values: embedded[idx],
      metadata: {
        text: chunk.pageContent,
        source: chunk.metadata.source,
        chunkIndex: chunk.metadata.chunkIndex,
      },
    }));

    await pineconeIndex.upsert(vectors);
  }

  console.log("✔ Upsert complete");
}

// ======================================================
// 5. AUTOMATIC NEW FILE INGESTION
// ======================================================

function getAllFilesRecursive(folder: string) {
  const results: string[] = [];

  function walk(dir: string) {
    const items = fs.readdirSync(dir);
    for (let item of items) {
      const full = path.join(dir, item);
      const stat = fs.statSync(full);

      if (stat.isDirectory()) walk(full);
      else if (/\.(txt|md|mdx|pdf)$/i.test(item)) results.push(full);
    }
  }

  walk(folder);
  return results;
}

async function checkForNewFilesAndIngest() {
  const files = getAllFilesRecursive(DATA_DIR);
  const newFiles = files.filter((f) => !processedFiles.has(f));

  if (newFiles.length === 0) {
    return;
  }

  console.log("📄 New files detected:");
  console.log(newFiles);

  // Load docs
  const docs: Document[] = [];
  for (let file of newFiles) {
    if (file.endsWith(".pdf")) {
      const buffer = fs.readFileSync(file);
      const pdfParse = (await import("pdf-parse")).default;
      const data = await pdfParse(buffer);
      docs.push(new Document({ pageContent: data.text, metadata: { source: file } }));
    } else {
      const text = fs.readFileSync(file, "utf-8");
      docs.push(new Document({ pageContent: text, metadata: { source: file } }));
    }
  }

  const chunks = await chunkDocuments(docs);
  await upsertToPinecone(chunks);

  // Save processed state
  newFiles.forEach((f) => processedFiles.add(f));
  saveProcessedFiles();

  console.log("🎉 New files indexed successfully!");
}

function startWatchingDataFolder() {
  console.log("👀 Watching /data folder for new files...");

  let debounce: any;
  fs.watch(DATA_DIR, { recursive: true }, () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => checkForNewFilesAndIngest(), 500);
  });
}

// ======================================================
// 6. RETRIEVAL CHAIN
// ======================================================

async function createRetrievalChain() {
  const vectorstore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
  });

  const retriever = vectorstore.asRetriever({ k: 4 });

  const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI agent representing Rishabh Gupta, a backend-focused full-stack developer. 
Your job is to answer queries with precision, clarity, and professionalism, based ONLY on 
the provided context (RAG documents) + universal common knowledge of software engineering.

== Core Rules ==
1. Always respond in first person AS Rishabh ("I", "my", "me").
2. ALWAYS stay professional, confident, and concise.
3. NEVER fabricate technical experiences or projects that are not in the provided context.
4. If the user asks something unrelated to development, career, skills, or projects,
   respond politely and steer the conversation back professionally.
5. If the answer is not present in the context, say:
   “I don’t have that information available in my current portfolio.”
6. Adapt the style to the intent of the question
   - If they ask for an introduction → give a professional intro.
   - If they ask technical questions → give technical/precise answers.
   - If they ask HR-style questions (e.g., Why should we hire you?) → answer like an interview.
   - If they ask general questions → respond politely but professionally.
7. If asked about project tell how much you know and let them know that rest projects they can explore in the project section of the portfolio.
== Tone ==
- Confident but not arrogant
- Professional but not robotic
- Clear, structured, trustworthy

== IMPORTANT ==
Do NOT introduce yourself unless the user explicitly asks for:
“introduce yourself”, “tell me about yourself”, “who are you”, etc.

== Output format ==
Provide the best possible helpful answer to the user’s question based on the above rules.


Context:
{context}

Question: {input}

`);

  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    apiKey: GOOGLE_API_KEY,
    temperature: 0,
  });

  const docChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const ragChain = await createLCRetrievalChain({
    retriever,
    combineDocsChain: docChain,
  });

  return ragChain;
}

// ======================================================
// 7. MAIN INITIALIZER
// ======================================================

export async function initializeRag() {
  console.log("\n====== INITIALIZING RISHABH'S RAG PIPELINE ======");

  embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: HF_API_KEY,
    model: EMBEDDING_MODEL,
  });

  await initializePinecone();

  const stats = await pineconeIndex.describeIndexStats();
  if (stats.totalRecordCount === 0) {
    const docs = await ingestDocuments();
    const chunks = await chunkDocuments(docs);
    await upsertToPinecone(chunks);

    // Mark initial files as processed
    getAllFilesRecursive(DATA_DIR).forEach(f => processedFiles.add(f));
    saveProcessedFiles();
  } else {
    console.log("✔ Existing data found, skipping initial ingestion");
  }

  // Start watching for new files
  startWatchingDataFolder();

  const chain = await createRetrievalChain();
  console.log("🚀 RAG is READY!");
  return chain;
}
