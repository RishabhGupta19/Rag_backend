import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path
import asyncio

# --- 1. FastAPI Initialization (MOVED UP) ---
# Initialize FastAPI FIRST so the port binding happens immediately
app = FastAPI(title="RAG Pipeline Query API")

# Allow CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Global Variables for RAG Components ---
QA_CHAIN = None
API_KEY = None
LOADING_STATUS = "not_started"  # Possible values: not_started, loading, loaded, failed

# --- 3. Background RAG Loading Function ---
def load_rag_pipeline_sync():
    """Load the RAG pipeline in the background."""
    global QA_CHAIN, API_KEY, LOADING_STATUS
    
    LOADING_STATUS = "loading"
    print("🔄 Loading RAG pipeline in background...", flush=True)
    
    try:
        print("🔄 Step 1: Attempting to import rag_pipeline...", flush=True)
        from rag_pipeline import QA_CHAIN as _QA_CHAIN, API_KEY as _API_KEY
        
        print("🔄 Step 2: Import successful, assigning to globals...", flush=True)
        QA_CHAIN = _QA_CHAIN
        API_KEY = _API_KEY
        
        LOADING_STATUS = "loaded"
        print("✅ RAG pipeline loaded successfully!", flush=True)
        print(f"✅ QA_CHAIN type: {type(QA_CHAIN)}", flush=True)
        print(f"✅ API_KEY set: {bool(API_KEY)}", flush=True)
        
    except ImportError as e:
        LOADING_STATUS = "failed"
        print(f"❌ IMPORT ERROR: Could not import rag_pipeline.", file=sys.stderr, flush=True)
        print(f"❌ Error details: {e}", file=sys.stderr, flush=True)
        print(f"❌ Error type: {type(e).__name__}", file=sys.stderr, flush=True)
        import traceback
        print(f"❌ Full traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        
    except Exception as e:
        LOADING_STATUS = "failed"
        print(f"❌ GENERAL ERROR: Error during RAG pipeline initialization.", file=sys.stderr, flush=True)
        print(f"❌ Error details: {e}", file=sys.stderr, flush=True)
        print(f"❌ Error type: {type(e).__name__}", file=sys.stderr, flush=True)
        import traceback
        print(f"❌ Full traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)

# --- 4. Startup Event - Trigger Background Loading ---
@app.on_event("startup")
async def startup_event():
    """Trigger RAG pipeline loading in background without blocking startup."""
    print("🚀 Server started!", flush=True)
    # TEMPORARILY DISABLED - Uncomment to enable RAG loading
    try:
        asyncio.create_task(asyncio.to_thread(load_rag_pipeline_sync))
    except Exception as e:
        print(f"❌ Failed to start background loading task: {e}", file=sys.stderr, flush=True)
        global LOADING_STATUS
        LOADING_STATUS = "failed"
    print("⚠️ RAG loading is DISABLED for debugging. Server will run but /query will not work.", flush=True)

# --- 5. Request Model ---
class QueryRequest(BaseModel):
    """Schema for the incoming user query."""
    question: str
    
# --- 6. Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint - responds immediately."""
    return {
        "message": "Server is running",
        "rag_status": LOADING_STATUS,
        "rag_initialized": QA_CHAIN is not None
    }

@app.get("/health")
async def health():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "rag_status": LOADING_STATUS
    }

@app.get("/status")
async def status():
    """Detailed status endpoint."""
    import os
    files_in_dir = os.listdir('.')
    
    return {
        "server": "running",
        "rag_pipeline": {
            "status": LOADING_STATUS,
            "available": QA_CHAIN is not None
        },
        "environment": {
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "files_present": files_in_dir,
            "rag_pipeline_exists": "rag_pipeline.py" in files_in_dir
        }
    }

@app.post("/query")
async def query_rag_pipeline(request: QueryRequest):
    """Accepts a question and uses the initialized RAG chain to get the answer."""
    
    if LOADING_STATUS == "loading":
        raise HTTPException(
            status_code=503, 
            detail="RAG service is still loading. Please try again in a moment."
        )
    
    if LOADING_STATUS == "failed":
        raise HTTPException(
            status_code=503, 
            detail="RAG service failed to initialize. Please check server logs."
        )
    
    if QA_CHAIN is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG service is not initialized. Please check server logs."
        )
         
    if not API_KEY or API_KEY == "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc":
        print("⚠️ WARNING: API Key issue detected. Attempting to proceed...", file=sys.stderr)
    
    print(f"📥 Received query: {request.question}")
    
    try:
        # Use the global QA_CHAIN (LangChain RAG)
        result = QA_CHAIN.invoke({"query": request.question})
        
        answer = result.get('result', "I'm sorry, I couldn't find the answer in the provided documents.")
        
        # Extract sources from the returned source_documents list
        sources = [
            {
                "uri": doc.metadata.get('source', 'N/A'), 
                # Use the source path as the title, removing path prefix
                "title": Path(doc.metadata.get('source', 'N/A')).name 
            }
            for doc in result.get('source_documents', [])
        ]
        
        print(f"✅ Query processed successfully")
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        print(f"❌ Error during RAG query: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal RAG Chain Error: {str(e)}")




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
