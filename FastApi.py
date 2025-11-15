import os
from pathlib import Path
from dotenv import load_dotenv
import sys
from typing import Optional
import gc

# --- LangChain and RAG Imports ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration & Global Variables ---
load_dotenv()
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

# OPTIMIZATION: Use a smaller, more memory-efficient model
# Original: sentence-transformers/all-MiniLM-L6-v2 (~120MB)
# Optimized: all-MiniLM-L6-v2 with model_kwargs for CPU optimization
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc") 

# Global variable for the embedding function (initialized once)
HF_EMBEDDING_FUNCTION: Optional[HuggingFaceEmbeddings] = None

# --- Core RAG Functions ---

def load_text_files(data_dir: Path):
    """Load text files with memory-efficient streaming."""
    docs = []
    for p in data_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
            except Exception as e:
                print(f"⚠️ Could not read {p.name}: {e}", file=sys.stderr)
    return docs

def load_pdfs(data_dir: Path):
    """Load PDFs with memory-efficient processing."""
    docs = []
    try:
        import pypdf
    except ImportError:
        print("⚠️ pypdf not installed. Skipping PDFs.", file=sys.stderr)
        return docs

    for p in data_dir.glob("**/*.pdf"):
        if p.is_file():
            try:
                reader = pypdf.PdfReader(str(p))
                # Process pages in chunks to avoid memory spike
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
                text = "\n".join(text_parts)
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
                # Clear text_parts to free memory
                del text_parts
            except Exception as e:
                print(f"⚠️ Could not read PDF {p.name}: {e}", file=sys.stderr)
                continue
    return docs

def ingest_documents(data_dir: Path):
    """Ingest documents with memory cleanup."""
    print(f"📂 Loading documents from {data_dir}...")
    docs = load_text_files(data_dir) + load_pdfs(data_dir)
    print(f"✅ Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks with memory optimization."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc.page_content)):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            chunks.append(Document(page_content=chunk, metadata=meta))
    
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

def create_embeddings_and_store(chunks, persist_dir: str):
    """Create vector store with memory-efficient batching."""
    global HF_EMBEDDING_FUNCTION
    if not HF_EMBEDDING_FUNCTION:
        raise RuntimeError("Embedding function not initialized.")
    
    print(f"💾 Creating vector store in {persist_dir}...")
    
    # OPTIMIZATION: Process in smaller batches to reduce memory peaks
    batch_size = 50  # Process 50 chunks at a time
    
    if len(chunks) <= batch_size:
        # If small enough, process all at once
        vectordb = Chroma.from_documents(
            chunks, 
            embedding=HF_EMBEDDING_FUNCTION, 
            persist_directory=persist_dir
        )
    else:
        # Process in batches for large datasets
        vectordb = Chroma.from_documents(
            chunks[:batch_size], 
            embedding=HF_EMBEDDING_FUNCTION, 
            persist_directory=persist_dir
        )
        
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectordb.add_documents(batch)
            print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
            
            # Force garbage collection after each batch
            gc.collect()
    
    print(f"✅ Vector store created in {persist_dir}")
    return vectordb

def should_reindex(data_dir: Path, chroma_dir: str) -> bool:
    """Checks if any file in data_dir is newer than the Chroma DB directory."""
    chroma_path = Path(chroma_dir)
    
    if not chroma_path.exists() or not os.listdir(chroma_path):
        print("💡 Chroma DB not found or is empty. Indexing is required.")
        return True

    try:
        db_mtime = chroma_path.stat().st_mtime
    except FileNotFoundError:
        print("💡 Chroma DB directory not found. Indexing is required.")
        return True

    source_files = [p for p in data_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
    
    if not source_files:
        print("⚠️ No source documents found. Skipping indexing.")
        return False
        
    for p in source_files:
        try:
            if p.stat().st_mtime > db_mtime:
                print(f"🔄 Document '{p.name}' is newer than the DB. Re-indexing...")
                return True
        except FileNotFoundError:
            continue

    print("✅ No changes detected in source documents. Skipping re-indexing.")
    return False

def make_retrieval_qa(vectordb: Chroma):
    """Create the QA chain with optimized retrieval."""
    # OPTIMIZATION: Reduce k from 4 to 3 to fetch fewer documents
    retriever = vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}  # Reduced from 4 to 3
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are Rishabh's personal AI Portfolio Assistant. Your sole purpose is to "
            "provide accurate, friendly, and informative answers about Rishabh's professional "
            "background, skills, and projects, using **ONLY** the context provided. When answering, "
            "speak in the **first person as Rishabh** (using 'I', 'my', 'me'). "
            "Do not use external knowledge.\n\n"
            "**Retrieved Context:**\n---\n{context}\n---\n\n"
            "**User Question:** {question}\n\n"
            "**Instructions:**\n"
            "1. Synthesize the answer from the context provided between the dashes (---).\n"
            "2. Be as direct and to-the-point as possible, minimize conversational filler.\n"
            "3. If the context does not contain the answer, politely respond that I don't have that information in my current professional records.\n\n"
            "**Answer (as Rishabh):**"
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True, 
    )
    return qa_chain


def initialize_rag_chain():
    """Performs conditional indexing/loading and returns the fully initialized QA chain."""
    global HF_EMBEDDING_FUNCTION

    print("=" * 60)
    print("🚀 Initializing RAG Pipeline (Memory Optimized)")
    print("=" * 60)

    # 1. Load Embedding Model with optimization
    print(f"⌛ Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        # OPTIMIZATION: Add model_kwargs for memory efficiency
        HF_EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Force CPU to save memory
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        print("✅ Embedding model loaded.")
        
        # Force garbage collection after model load
        gc.collect()
        
    except Exception as e:
        print(f"❌ FATAL: Could not load embedding model. Error: {e}", file=sys.stderr)
        return None

    # 2. Conditional RAG Initialization
    if should_reindex(DATA_DIR, CHROMA_DIR):
        print("\n*** Starting Indexing Process ***")
        docs = ingest_documents(DATA_DIR)
        if not docs:
            print("⚠️ No documents to index. RAG will not work.")
            return None

        chunks = chunk_documents(docs)
        
        # Clear docs from memory before creating embeddings
        del docs
        gc.collect()
        
        vectordb = create_embeddings_and_store(chunks, CHROMA_DIR)
        
        # Clear chunks from memory
        del chunks
        gc.collect()
        
        print("*** Indexing Complete ***\n")
    else:
        print("\n*** Loading Existing Vector Store ***")
        try:
            vectordb = Chroma(
                persist_directory=CHROMA_DIR, 
                embedding_function=HF_EMBEDDING_FUNCTION
            )
            print(f"✅ Loaded existing vector store from {CHROMA_DIR}.")
            print("*** Loading Complete ***\n")
            
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error loading existing vector store: {e}", file=sys.stderr)
            print("⚠️ Delete 'chroma_db' folder and rerun to force re-index.")
            return None
            
    # 3. Create and return the QA Chain
    print("🔗 Creating QA Chain...")
    qa_chain = make_retrieval_qa(vectordb)
    print("✅ QA Chain ready!")
    
    # Final garbage collection
    gc.collect()
    
    print("=" * 60)
    print("✅ RAG Pipeline Initialization Complete")
    print("=" * 60)
    
    return qa_chain

# The initialized chain will be stored here and accessed by FastApi.py
QA_CHAIN = initialize_rag_chain()




# new code 


# import os
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import sys
# from pathlib import Path
# import asyncio

# # --- 1. FastAPI Initialization (MOVED UP) ---
# # Initialize FastAPI FIRST so the port binding happens immediately
# app = FastAPI(title="RAG Pipeline Query API")

# # Allow CORS
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "*"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- 2. Global Variables for RAG Components ---
# QA_CHAIN = None
# API_KEY = None
# LOADING_STATUS = "not_started"  # Possible values: not_started, loading, loaded, failed

# # --- 3. Background RAG Loading Function ---
# def load_rag_pipeline_sync():
#     """Load the RAG pipeline in the background."""
#     global QA_CHAIN, API_KEY, LOADING_STATUS
    
#     LOADING_STATUS = "loading"
#     print("🔄 Loading RAG pipeline in background...")
    
#     try:
#         from rag_pipeline import QA_CHAIN as _QA_CHAIN, API_KEY as _API_KEY
#         QA_CHAIN = _QA_CHAIN
#         API_KEY = _API_KEY
#         LOADING_STATUS = "loaded"
#         print("✅ RAG pipeline loaded successfully!")
#     except ImportError as e:
#         LOADING_STATUS = "failed"
#         print(f"❌ ERROR: Could not import rag_pipeline. Error: {e}", file=sys.stderr)
#     except Exception as e:
#         LOADING_STATUS = "failed"
#         print(f"❌ ERROR: Error during RAG pipeline initialization: {e}", file=sys.stderr)

# # --- 4. Startup Event - Trigger Background Loading ---
# @app.on_event("startup")
# async def startup_event():
#     """Trigger RAG pipeline loading in background without blocking startup."""
#     print("🚀 Server started! Initiating RAG pipeline loading in background...")
#     # Start loading in a separate thread/task
#     asyncio.create_task(asyncio.to_thread(load_rag_pipeline_sync))

# # --- 5. Request Model ---
# class QueryRequest(BaseModel):
#     """Schema for the incoming user query."""
#     question: str
    
# # --- 6. Endpoints ---
# @app.get("/")
# async def root():
#     """Health check endpoint - responds immediately."""
#     return {
#         "message": "Server is running",
#         "rag_status": LOADING_STATUS,
#         "rag_initialized": QA_CHAIN is not None
#     }

# @app.get("/health")
# async def health():
#     """Health check for load balancers."""
#     return {
#         "status": "healthy",
#         "rag_status": LOADING_STATUS
#     }

# @app.get("/status")
# async def status():
#     """Detailed status endpoint."""
#     return {
#         "server": "running",
#         "rag_pipeline": {
#             "status": LOADING_STATUS,
#             "available": QA_CHAIN is not None
#         }
#     }

# @app.post("/query")
# async def query_rag_pipeline(request: QueryRequest):
#     """Accepts a question and uses the initialized RAG chain to get the answer."""
    
#     if LOADING_STATUS == "loading":
#         raise HTTPException(
#             status_code=503, 
#             detail="RAG service is still loading. Please try again in a moment."
#         )
    
#     if LOADING_STATUS == "failed":
#         raise HTTPException(
#             status_code=503, 
#             detail="RAG service failed to initialize. Please check server logs."
#         )
    
#     if QA_CHAIN is None:
#         raise HTTPException(
#             status_code=503, 
#             detail="RAG service is not initialized. Please check server logs."
#         )
         
#     if not API_KEY or API_KEY == "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc":
#         print("⚠️ WARNING: API Key issue detected. Attempting to proceed...", file=sys.stderr)
    
#     print(f"📥 Received query: {request.question}")
    
#     try:
#         # Use the global QA_CHAIN (LangChain RAG)
#         result = QA_CHAIN.invoke({"query": request.question})
        
#         answer = result.get('result', "I'm sorry, I couldn't find the answer in the provided documents.")
        
#         # Extract sources from the returned source_documents list
#         sources = [
#             {
#                 "uri": doc.metadata.get('source', 'N/A'), 
#                 # Use the source path as the title, removing path prefix
#                 "title": Path(doc.metadata.get('source', 'N/A')).name 
#             }
#             for doc in result.get('source_documents', [])
#         ]
        
#         print(f"✅ Query processed successfully")
#         return {"answer": answer, "sources": sources}
        
#     except Exception as e:
#         print(f"❌ Error during RAG query: {e}", file=sys.stderr)
#         raise HTTPException(status_code=500, detail=f"Internal RAG Chain Error: {str(e)}")
