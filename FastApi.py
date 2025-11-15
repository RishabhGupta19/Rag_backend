# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import sys
# from pathlib import Path

# # --- 1. Import RAG Core Components ---
# # NOTE: This line triggers the execution of rag_pipeline.py, 
# # which loads the models and initializes the QA_CHAIN globally.
# try:
#     from rag_pipeline import QA_CHAIN, API_KEY 
# except ImportError as e:
#     print(f"❌ FATAL: Could not import rag_pipeline. Ensure 'rag_pipeline.py' is in the same directory. Error: {e}", file=sys.stderr)
#     sys.exit(1)
# except Exception as e:
#     print(f"❌ FATAL: Error during RAG pipeline initialization: {e}", file=sys.stderr)
#     sys.exit(1)


# # --- 2. FastAPI Initialization ---
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

# # --- 3. Request Model ---
# class QueryRequest(BaseModel):
#     """Schema for the incoming user query."""
#     question: str
    
# # --- 4. Endpoint ---
# @app.get("/")
# async def root():
#     return {"message": "Server is running"}
# @app.post("/query")



# async def query_rag_pipeline(request: QueryRequest):
#     """Accepts a question and uses the initialized RAG chain to get the answer."""
    
#     if QA_CHAIN is None:
#          raise HTTPException(status_code=503, detail="RAG service is not initialized or failed to load vector store.")
         
#     if not API_KEY or API_KEY == "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc":
#          # In a production scenario, you might deny service here, but for development
#          print("❌ WARNING: API Key issue detected. Attempting to proceed...", file=sys.stderr)

#     print(f"Received query: {request.question}")
    
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

#         return {"answer": answer, "sources": sources}

#     except Exception as e:
#         print(f"An unexpected error occurred during RAG query: {e}", file=sys.stderr)
#         raise HTTPException(status_code=500, detail=f"Internal RAG Chain Error: {str(e)}")




# new code 


import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

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

# --- 3. Startup Event - Load RAG Pipeline AFTER Server Starts ---
@app.on_event("startup")
async def load_rag_pipeline():
    """Load the RAG pipeline after FastAPI has bound to the port."""
    global QA_CHAIN, API_KEY
    
    print("🚀 Server starting... Loading RAG pipeline...")
    
    try:
        from rag_pipeline import QA_CHAIN as _QA_CHAIN, API_KEY as _API_KEY
        QA_CHAIN = _QA_CHAIN
        API_KEY = _API_KEY
        print("✅ RAG pipeline loaded successfully!")
    except ImportError as e:
        print(f"⚠️ WARNING: Could not import rag_pipeline. Ensure 'rag_pipeline.py' is in the same directory. Error: {e}", file=sys.stderr)
        print("⚠️ Server will run but /query endpoint will return 503", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ WARNING: Error during RAG pipeline initialization: {e}", file=sys.stderr)
        print("⚠️ Server will run but /query endpoint will return 503", file=sys.stderr)

# --- 4. Request Model ---
class QueryRequest(BaseModel):
    """Schema for the incoming user query."""
    question: str
    
# --- 5. Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint - responds immediately."""
    return {
        "message": "Server is running",
        "rag_initialized": QA_CHAIN is not None
    }

@app.get("/health")
async def health():
    """Health check for load balancers."""
    return {
        "status": "healthy",
        "rag_available": QA_CHAIN is not None
    }

@app.post("/query")
async def query_rag_pipeline(request: QueryRequest):
    """Accepts a question and uses the initialized RAG chain to get the answer."""
    
    if QA_CHAIN is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG service is not initialized or failed to load vector store. Please check server logs."
        )
         
    if not API_KEY or API_KEY == "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc":
        # In a production scenario, you might deny service here, but for development
        print("❌ WARNING: API Key issue detected. Attempting to proceed...", file=sys.stderr)
    
    print(f"Received query: {request.question}")
    
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
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"An unexpected error occurred during RAG query: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal RAG Chain Error: {str(e)}")
