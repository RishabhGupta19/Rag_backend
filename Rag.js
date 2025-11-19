// import { config } from "dotenv";
// import fs from "fs";
// import path from "path";
// import { Document } from "@langchain/core/documents";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
// import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
// import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { createRetrievalChain as createLCRetrievalChain } from "langchain/chains/retrieval";
// import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
// import { StringOutputParser } from "@langchain/core/output_parsers";
// // @ts-ignore


// if (import.meta.url === `file://${process.argv[1]}`) {
//   // your startup code
// }
// config();

// // ==================== CONFIG & VALIDATION ====================
// const DATA_DIR = "data";
// const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "rishabh-portfolio";
// const HF_API_KEY = process.env.HF_API_KEY;
// const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

// // ==================== CONFIG & VALIDATION ====================
// console.log("Loading .env variables...");

// const requiredEnv = {
//   PINECONE_API_KEY: process.env.PINECONE_API_KEY,
//   HF_API_KEY: process.env.HF_API_KEY,
//   GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
// };

// const missing = Object.entries(requiredEnv)
//   .filter(([, value]) => !value)
//   .map(([key]) => key);

// if (missing.length > 0) {
//   console.error("MISSING REQUIRED ENVIRONMENT VARIABLES:");
//   missing.forEach(key => console.error(`   ✗ ${key}=${requiredEnv[key as keyof typeof requiredEnv] || "(empty)"}`));
//   console.error("\nPlease add them to your .env file:");
//   console.error("Example .env:");
//   console.error("PINECONE_API_KEY=pcsk_...");
//   console.error("HF_API_KEY=hf_...");
//   console.error("GOOGLE_API_KEY=AIza...");
//   process.exit(1);
// }

// console.log("All environment variables loaded successfully!");
// const EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
// const EMBEDDING_DIMENSION = 384;

// let embeddings: HuggingFaceInferenceEmbeddings;
// let pineconeClient: PineconeClient;
// let pineconeIndex: any;

// // ==================== 1. LOAD DOCUMENTS ====================
// async function loadTextFiles(dir: string): Promise<Document[]> {
//   const docs: Document[] = [];
//   let items;
//   try {
//     items = fs.readdirSync(dir);
//     console.log(`Found ${items.length} items in ${dir}`);
//   } catch (err) {
//     console.error(`Failed to read directory: ${dir}`, err);
//     return docs;
//   }

//   for (const item of items) {
//     const fullPath = path.join(dir, item);
//     let stat;
//     try {
//       stat = fs.statSync(fullPath);
//     } catch (err) {
//       console.error(`Cannot stat file: ${fullPath}`, err);
//       continue;
//     }

//     if (stat.isDirectory()) {
//       console.log(`Recursing into directory: ${fullPath}`);
//       docs.push(...await loadTextFiles(fullPath));
//       continue;
//     }

//     if (item.endsWith(".txt") || item.endsWith(".md") || item.endsWith(".mdx")) {
//       try {
//         const content = fs.readFileSync(fullPath, "utf-8");
//         docs.push(new Document({ pageContent: content, metadata: { source: fullPath } }));
//         console.log(`Loaded text file: ${item} (${content.length} chars)`);
//       } catch (err) {
//         console.error(`Failed to read text file ${item}:`, err);
//       }
//     }
//   }
//   return docs;
// }

// async function loadPDFs(dir: string): Promise<Document[]> {
//   const docs: Document[] = [];
//   let items;
//   try {
//     items = fs.readdirSync(dir);
//   } catch (err) {
//     console.error(`Failed to read directory for PDFs: ${dir}`, err);
//     return docs;
//   }

//   for (const item of items) {
//     const fullPath = path.join(dir, item);
//     let stat;
//     try {
//       stat = fs.statSync(fullPath);
//     } catch (err) {
//       console.error(`Cannot stat PDF candidate: ${fullPath}`, err);
//       continue;
//     }

//     if (stat.isDirectory()) {
//       docs.push(...await loadPDFs(fullPath));
//       continue;
//     }

//     if (item.toLowerCase().endsWith(".pdf")) {
//       try {
//         const buffer = fs.readFileSync(fullPath);
//         const pdfParse = (await import("pdf-parse")).default;
//         const data = await pdfParse(buffer);
//         docs.push(new Document({ pageContent: data.text, metadata: { source: fullPath } }));
//         console.log(`Loaded PDF: ${item} (${data.text.length} chars extracted)`);
//       } catch (err) {
//         console.error(`Failed to parse PDF ${item}:`, err);
//       }
//     }
//   }
//   return docs;
// }

// async function ingestDocuments(): Promise<Document[]> {
//   console.log("Starting document ingestion...");
//   try {
//     const [textDocs, pdfDocs] = await Promise.all([
//       loadTextFiles(DATA_DIR).catch(err => {
//         console.error("loadTextFiles failed:", err);
//         return [];
//       }),
//       loadPDFs(DATA_DIR).catch(err => {
//         console.error("loadPDFs failed:", err);
//         return [];
//       })
//     ]);

//     const allDocs = [...textDocs, ...pdfDocs];
//     console.log(`Successfully loaded ${allDocs.length} documents (${textDocs.length} text + ${pdfDocs.length} PDFs)`);
//     return allDocs;
//   } catch (err) {
//     console.error("ingestDocuments completely failed:", err);
//     return [];
//   }
// }

// // ==================== 2. CHUNKING ====================
// async function chunkDocuments(docs: Document[]): Promise<Document[]> {
//   console.log(`Chunking ${docs.length} documents...`);
//   try {
//     const splitter = new RecursiveCharacterTextSplitter({
//       chunkSize: 800,
//       chunkOverlap: 100,
//     });

//     const chunks = await splitter.splitDocuments(docs);
//     chunks.forEach((chunk, i) => {
//       chunk.metadata.chunkIndex = i;
//     });

//     console.log(`Successfully created ${chunks.length} chunks`);
//     return chunks;
//   } catch (err) {
//     console.error("Chunking failed:", err);
//     return [];
//   }
// }

// // ==================== 3. PINECONE INIT ====================
// async function initializePinecone() {
//   console.log("Initializing Pinecone client...");

//   if (!process.env.PINECONE_API_KEY) {
//     throw new Error("PINECONE_API_KEY is missing in .env");
//   }

//   try {
//     pineconeClient = new PineconeClient(); // Auto-uses env var
//     console.log("Pinecone client created");
//   } catch (err) {
//     console.error("Failed to create Pinecone client:", err);
//     throw err;
//   }

//   let indexesResponse;
//   try {
//     console.log("Listing existing indexes...");
//     indexesResponse = await pineconeClient.listIndexes();
//     const indexNames = indexesResponse.indexes?.map((idx: any) => idx.name) || [];
//     console.log(`Found ${indexNames.length} indexes:`, indexNames);
    
//     if (!indexNames.includes(PINECONE_INDEX_NAME)) {
//       console.log(`Creating new index: ${PINECONE_INDEX_NAME}`);
//       await pineconeClient.createIndex({
//         name: PINECONE_INDEX_NAME,
//         dimension: EMBEDDING_DIMENSION,
//         metric: "cosine",
//         spec: {
//           serverless: {
//             cloud: "aws" as const,
//             region: "us-east-1" as const,
//           },
//         },
//       });
//       console.log("Index creation requested. Waiting 45s for initialization...");
//       await new Promise(r => setTimeout(r, 45000));
//     } else {
//       console.log(`Index ${PINECONE_INDEX_NAME} already exists`);
//     }
//   } catch (err) {
//     console.error("Pinecone index check/create failed:", err);
//     throw err;
//   }

//   try {
//     pineconeIndex = pineconeClient.index(PINECONE_INDEX_NAME);
//     console.log("Successfully connected to Pinecone index");
//   } catch (err) {
//     console.error("Failed to get index handle:", err);
//     throw err;
//   }
// }

// // ==================== 4. UPSERT ====================
// async function upsertToPinecone(chunks: Document[]) {
//   if (chunks.length === 0) {
//     console.log("No chunks to upsert");
//     return;
//   }

//   console.log(`Upserting ${chunks.length} chunks in batches...`);

//   const batchSize = 100;
//   const batches = [];

//   for (let i = 0; i < chunks.length; i += batchSize) {
//     const batch = chunks.slice(i, i + batchSize);
//     const texts = batch.map(c => c.pageContent);

//     let embeddingsArray;
//     try {
//       embeddingsArray = await embeddings.embedDocuments(texts);
//       console.log(`Embedded batch ${i / batchSize + 1}/${Math.ceil(chunks.length / batchSize)}`);
//     } catch (err) {
//       console.error(`Embedding failed for batch starting at ${i}:`, err);
//       continue;
//     }

//     const vectors = batch.map((chunk, idx) => ({
//       id: `chunk-${i + idx}`,
//       values: embeddingsArray[idx],
//       metadata: {
//         text: chunk.pageContent,
//         source: chunk.metadata.source,
//         chunkIndex: chunk.metadata.chunkIndex,
//       }
//     }));

//     batches.push(vectors);
//   }

//   try {
//     await Promise.all(
//       batches.map((batch, batchIdx) =>
//         pineconeIndex.upsert(batch)
//           .then(() => console.log(`Upserted batch ${batchIdx + 1}/${batches.length}`))
//           .catch((err: any) => console.error(`Upsert failed for batch ${batchIdx + 1}:`, err))
//       )
//     );
//     console.log(`All ${chunks.length} vectors upserted successfully!`);
//   } catch (err) {
//     console.error("Some upserts failed:", err);
//   }
// }

// // ==================== 5. RETRIEVAL CHAIN ====================

// async function createRetrievalChain() {
//   console.log("Creating retrieval chain...");

//   // Build LangChain vectorstore wrapper
//   const vectorstore = await PineconeStore.fromExistingIndex(
//     embeddings,
//     {
//       pineconeIndex,
//       namespace: undefined,   // or your namespace if you used one
//     }
//   );

//   const retriever = vectorstore.asRetriever({ k: 4 });
//   console.log("Retriever created using PineconeStore");

//   const prompt = ChatPromptTemplate.fromTemplate(`
// "You are Rishabh's personal AI Portfolio Assistant. Your sole purpose is to "
//             "provide accurate, friendly, and informative answers about Rishabh's professional "
//             "background, skills, and projects, using **ONLY** the context provided. When answering, "
//             "speak in the **first person as Rishabh** (using 'I', 'my', 'me'). "
//             "Do not use external knowledge.\n\n"
// {context}

// Question: {input}

// "1. Synthesize the answer from the context provided between the dashes (---).\n"
//             "2. Be as direct and to-the-point as possible, minimize conversational filler.\n"
//             "3. If the context does not contain the answer, politely respond that I don't have that information in my current professional records.\n\n"
//              "4. If asked for intoduction tell it  in a professional way"
//              "5." Do NOT include any explanations, summaries, or commentary outside of the reply.
//             "**Answer (as Rishabh):**"
// `);

//   const llm = new ChatGoogleGenerativeAI({
//     model: "gemini-2.0-flash",
//     apiKey: GOOGLE_API_KEY,
//     temperature: 0,
//   });

//   const documentChain = await createStuffDocumentsChain({
//     llm,
//     prompt,
//     outputParser: new StringOutputParser(),
//   });

//   const ragChain = await createLCRetrievalChain({
//     retriever,
//     combineDocsChain: documentChain,
//   });

//   console.log("RAG chain created successfully!");
//   return ragChain;
// }

// // ==================== 6. MAIN INIT ====================
// export async function initializeRag() {
//   console.log("=".repeat(50));
//   console.log("INITIALIZING RISHABH'S RAG PIPELINE");
//   console.log("=".repeat(50));

//   try {
//     // 1. Embeddings
//     console.log("Initializing HuggingFace embeddings...");
//     embeddings = new HuggingFaceInferenceEmbeddings({
//       apiKey: HF_API_KEY,
//       model: EMBEDDING_MODEL,
//     });
//     console.log("Embeddings ready");

//     // 2. Pinecone
//     await initializePinecone();

//     // 3. Check & ingest
//     console.log("Checking vector count...");
//     const stats = await pineconeIndex.describeIndexStats();
//     console.log(`Index has ${stats.totalRecordCount || 0} vectors`);

//     if ((stats.totalRecordCount || 0) === 0) {
//       console.log("No data found → Starting ingestion...");
//       const docs = await ingestDocuments();
//       if (docs.length === 0) {
//         console.warn("No documents loaded — skipping chunking/upsert");
//       } else {
//         const chunks = await chunkDocuments(docs);
//         await upsertToPinecone(chunks);
//       }
//     } else {
//       console.log(`Found existing data — skipping ingestion`);
//     }

//     // 4. Chain
//     const chain = await createRetrievalChain();

//     console.log("RAG PIPELINE FULLY READY!");
//     return chain;
//   } catch (err) {
//     console.error("FATAL: RAG initialization failed:", err);
//     throw err;
//   }
// }


// // ==================== SCRIPT STARTUP ====================
// // -------------------- startup + interactive loop --------------------
// import readline from "readline";
// import { fileURLToPath } from "url";

// async function runInteractiveQueryLoop(chain: any) {
//   const rl = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout,
//     prompt: "You> ",
//   });

//   console.log("\nInteractive RAG console. Type a question or 'exit' to quit.\n");
//   rl.prompt();

//   rl.on("line", async (line) => {
//     const q = line.trim();
//     if (!q) {
//       rl.prompt();
//       return;
//     }
//     if (q.toLowerCase() === "exit" || q.toLowerCase() === "quit") {
//       console.log("Bye 👋");
//       rl.close();
//       process.exit(0);
//     }

//     try {
//       // chain might be a Runnable-like object. Many langchain chains expose .invoke()
//       // or .call(). Try invoke() first then call() as a fallback.
//       let result: any;
//       if (typeof chain.invoke === "function") {
//         result = await chain.invoke({ input: q });
//       } else if (typeof chain.call === "function") {
//         // some older chain APIs use call({ input })
//         result = await chain.call({ input: q });
//       } else if (typeof chain.run === "function") {
//         // some chains allow run(input)
//         result = await chain.run(q);
//       } else {
//         throw new Error("Chain object does not expose invoke/call/run. Inspect chain API.");
//       }

//       // result shape may vary. Commonly: { text } or { answer } or { context, answer }.
//       // Print intelligently:
//       if (result == null) {
//         console.log("No result (null/undefined).");
//       } else if (typeof result === "string") {
//         console.log("\nAssistant>", result, "\n");
//       } else if (typeof result === "object") {
//         // Try common keys
//         const text = result.answer ?? result.text ?? result.output ?? result.result ?? JSON.stringify(result);
//         console.log("\nAssistant>", text, "\n");

//         // If chain returns source documents, show them briefly
//         const ctx = result.context ?? result.sourceDocuments ?? result.source_docs ?? result.documents;
//         if (Array.isArray(ctx) && ctx.length > 0) {
//           console.log("Sources:");
//           for (let i = 0; i < Math.min(3, ctx.length); i++) {
//             const doc = ctx[i];
//             const src = doc.metadata?.source ?? doc.metadata?.sourceFile ?? doc.metadata?.filename ?? doc.metadata;
//             console.log(`  - ${src}`);
//           }
//           console.log("");
//         }
//       } else {
//         console.log("Assistant (raw):", result);
//       }
//     } catch (err) {
//       console.error("Query failed:", err);
//     }

//     rl.prompt();
//   });

//   rl.on("close", () => {
//     console.log("\nInteractive session closed.");
//     process.exit(0);
//   });
// }

// // Robust "is this file being run directly?" check that works with tsx/ts-node on Windows/Linux
// function isMainModule() {
//   const scriptPath = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
//   const thisFilePath = fileURLToPath(import.meta.url);
//   return scriptPath && path.resolve(scriptPath) === path.resolve(thisFilePath);
// }

// // If run directly, initialize and start the loop
// if (isMainModule()) {
//   (async () => {
//     try {
//       const chain = await initializeRag();
//       // optionally store chain globally, or pass to functions
//       await runInteractiveQueryLoop(chain);
//     } catch (err) {
//       console.error("FATAL: startup failed:", err);
//       process.exit(1);
//     }
//   })();
// }


// ======================================================
// RISHABH’S FULL RAG PIPELINE WITH AUTO-INGEST WATCHER
// ======================================================

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
