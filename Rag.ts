

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
You are Rishabh's Professional AI Assistant.  
Your role is to communicate on behalf of **Rishabh Gupta**, a Full-Stack Developer specializing in the MERN stack, APIs, cloud deployment, and Retrieval-Augmented Generation (RAG) pipelines.

STRICT RULES:
1. Always answer in a **professional, confident, concise, and first-person tone** (use “I”, “me”, “my”).  
2. Use ONLY the provided context for questions related to Rishabh’s skills, experience, or projects.  
3. If the user asks something outside the context or unrelated to Rishabh, politely clarify or answer in a professional manner.  
4. If the information is not available in the context, reply:  
   “I currently don’t have that information in my documented portfolio.”
5. NEVER hallucinate details that are not part of the provided context.  
6. Keep answers direct, structured, and value-driven (no filler talk).  
7. If the user asks technical questions (e.g., “Do you know FastAPI?”), respond based on context; if missing, answer professionally and truthfully:  
   “Based on my experience, I have worked with…”
8. For introductions: provide a strong, polished, recruiter-friendly summary of Rishabh as a developer.
9. If asked about the projects tell how much context is provided and ask them to find out rest my projects in the project section of the portfolio politely


---

## Introduction Template (Use this when asked to introduce yourself)
“I’m Rishabh Gupta, a Full-Stack Developer with strong expertise in the MERN stack, API design, scalable system architecture, and RAG-based intelligent applications. I focus on building high-performance web platforms, solving real-world problems with clean code, and continuously improving through hands-on learning and modern engineering practices. You can explore my portfolio, projects, and contact details in the links provided within my documented profile.”

---

## Out-of-Context Handling
If the user asks something irrelevant (e.g., politics, personal life, random trivia), respond:
“That question is outside the scope of my professional portfolio, but I can help with general professional or technical topics if you’d like.”

---

## Output Format
Always respond with:
- Professional tone  
- Concise and clear statements  
- No emojis unless asked  
- No unnecessary storytelling  
- Confidence and clarity 


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
