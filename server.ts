import express from "express";
import cors from "cors";
import { initializeRag } from "./Rag.ts";   // ✔ Correct import for TS + Render
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors({ origin: "*" }));

// -------------------- GLOBALS --------------------
let RAG_CHAIN: any = null;
let LOADING_STATUS = "not_started"; 
// values = "not_started" | "loading" | "loaded" | "failed"

console.log("🚀 Server Starting...");
console.log("⏳ Loading RAG pipeline in background...");

// -------------------- BACKGROUND RAG LOADING --------------------
(async () => {
  LOADING_STATUS = "loading";
  try {
    RAG_CHAIN = await initializeRag();
    LOADING_STATUS = "loaded";
    console.log("✅ RAG Pipeline Loaded Successfully");
  } catch (err) {
    LOADING_STATUS = "failed";
    console.error("❌ RAG Pipeline Failed to Initialize:", err);
  }
})();

// -------------------- ROUTES --------------------

// Basic Health Check
app.get("/", (req, res) => {
  res.json({
    message: "Server is running",
    rag_status: LOADING_STATUS,
    rag_initialized: RAG_CHAIN !== null
  });
});

// Status Endpoint
app.get("/status", (req, res) => {
  res.json({
    server: "running",
    rag: {
      status: LOADING_STATUS,
      initialized: RAG_CHAIN !== null
    }
  });
});

// ----------------------------------------------
// POST /query → Ask the RAG Chain a question
// ----------------------------------------------
app.post("/query", async (req, res) => {
  try {
    if (LOADING_STATUS === "loading")
      return res.status(503).json({ error: "RAG is still loading, wait a bit." });

    if (LOADING_STATUS === "failed")
      return res.status(503).json({ error: "RAG failed to initialize." });

    if (!RAG_CHAIN)
      return res.status(503).json({ error: "RAG is not ready yet." });

    const { question } = req.body;
    if (!question)
      return res.status(400).json({ error: "Missing 'question' in body" });

    console.log("📥 Incoming Query:", question);

    let result;

    if (typeof RAG_CHAIN.invoke === "function") {
      result = await RAG_CHAIN.invoke({ input: question });
    } else if (typeof RAG_CHAIN.call === "function") {
      result = await RAG_CHAIN.call({ input: question });
    } else {
      throw new Error("RAG chain has no invoke/call method");
    }

    const answer =
      result.answer ||
      result.text ||
      result.output ||
      "I could not find the answer in your documents.";

    const sources =
      (result.context || [])
        .map((doc: any) => ({
          source: doc.metadata?.source || "unknown",
          chunkIndex: doc.metadata?.chunkIndex
        }));

    return res.json({ answer, sources });

  } catch (err: any) {
    console.error("❌ Error during query:", err);
    return res.status(500).json({ error: "Internal RAG error: " + err.message });
  }
});

// -------------------- START SERVER --------------------
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🔥 RAG API running at http://localhost:${PORT}`);
});
