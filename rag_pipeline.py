
# from pathlib import Path
# import os
# from dotenv import load_dotenv

# # ✅ Load environment variables from .env
# load_dotenv()

# # ✅ LangChain imports (compatible with 1.x)
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import PromptTemplate
# from langchain_classic.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI


# # --- CONFIG ---
# DATA_DIR = Path("data")
# CHROMA_DIR = "chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



# # --- INGESTION + CHUNKING ---
# # (Keep load_text_files, load_pdfs, ingest_documents, chunk_documents, create_embeddings_and_store as they are)

# def load_text_files(data_dir: Path):
#     docs = []
#     for p in data_dir.glob("**/*"):
#         if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
#             text = p.read_text(encoding="utf-8", errors="ignore")
#             docs.append(Document(page_content=text, metadata={"source": str(p)}))
#     return docs


# def load_pdfs(data_dir: Path):
#     docs = []
#     try:
#         import pypdf
#     except ImportError:
#         print("⚠️ Please install pypdf to load PDFs: pip install pypdf")
#         return docs

#     for p in data_dir.glob("**/*.pdf"):
#         # Ensure the file exists before trying to read it
#         if p.is_file():
#             reader = pypdf.PdfReader(str(p))
#             text = "\n".join(page.extract_text() or "" for page in reader.pages)
#             docs.append(Document(page_content=text, metadata={"source": str(p)}))
#     return docs


# def ingest_documents(data_dir: Path):
#     docs = load_text_files(data_dir) + load_pdfs(data_dir)
#     print(f"✅ Loaded {len(docs)} documents.")
#     return docs


# def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     chunks = []
#     for doc in docs:
#         for i, chunk in enumerate(splitter.split_text(doc.page_content)):
#             meta = dict(doc.metadata)
#             meta["chunk_index"] = i
#             chunks.append(Document(page_content=chunk, metadata=meta))
#     print(f"✅ Split into {len(chunks)} chunks.")
#     return chunks


# def create_embeddings_and_store(chunks, persist_dir: str):
#     hf_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     vectordb = Chroma.from_documents(chunks, embedding=hf_emb, persist_directory=persist_dir)
#     print(f"✅ Vector store ready in {persist_dir}")
#     return vectordb

# # --- INDEXING CHECK ---
# def should_reindex(data_dir: Path, chroma_dir: str) -> bool:
#     """Checks if any file in data_dir is newer than the Chroma DB directory."""
#     chroma_path = Path(chroma_dir)
    
#     # 1. Check if Chroma DB directory exists
#     if not chroma_path.exists() or not os.listdir(chroma_path):
#         print("💡 Chroma DB not found or is empty. Indexing is required.")
#         return True

#     # Get the last modification time of the Chroma DB directory
#     # Note: On some systems, this may not be precise enough, but it's a common heuristic.
#     try:
#         db_mtime = chroma_path.stat().st_mtime
#     except FileNotFoundError:
#         # Should be covered by the check above, but for safety
#         print("💡 Chroma DB directory not found. Indexing is required.")
#         return True

#     # 2. Check document modification times
#     source_files = [p for p in data_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
    
#     if not source_files:
#         print("⚠️ No new documents found. Skipping indexing.")
#         return False
        
#     for p in source_files:
#         try:
#             # Check if the document's modification time is newer than the DB's
#             if p.stat().st_mtime > db_mtime:
#                 print(f"🔄 Document '{p.name}' is newer than the DB. Re-indexing...")
#                 return True
#         except FileNotFoundError:
#             # Handle case where a file disappears during the check
#             continue

#     print("✅ No changes detected in source documents. Skipping ingestion/indexing.")
#     return False

# # --- RETRIEVAL + QA ---
# def make_retrieval_qa(vectordb):
#     retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=(
#             "You are an expert document analysis assistant. Your sole purpose is to "
#             "generate a factual, concise, and direct answer based **ONLY** on the "
#             "following retrieved context. Do not use external knowledge.\n\n"
#             "**Retrieved Context:**\n---\n{context}\n---\n\n"
#             "**User Question:** {question}\n\n"
#             "**Instructions:**\n"
#             "1. Synthesize the answer from the context provided between the dashes (---).\n"
#             "2. Be as direct and to-the-point as possible, minimize conversational filler.\n"
#             "3. If the context does not contain the answer, you answer politely in positive manner and based on provided documents and never let them know that you answer based on a provided document.'\n\n"
#             "**Answer:**"
#         ),
#     )

#     # ✅ Initialize Gemini chat model (no manual API handling)
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         api_key="AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc",
#         temperature=0
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt},
#     )
#     return qa_chain


# # --- MAIN EXECUTION ---
# def main():
#     # 1. Check if re-indexing is needed
#     if should_reindex(DATA_DIR, CHROMA_DIR):
#         # 2. Run Ingestion, Chunking, and Embedding (if needed)
#         docs = ingest_documents(DATA_DIR)
#         if not docs:
#             print("⚠️ No documents found in 'data' folder! Cannot create vector store.")
#             return

#         chunks = chunk_documents(docs)
#         vectordb = create_embeddings_and_store(chunks, CHROMA_DIR)
#     else:
#         # 3. Load existing Vector Store (if no re-indexing needed)
#         try:
#             hf_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#             # The Chroma DB persistence logic automatically loads from the directory
#             vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=hf_emb)
#             print(f"✅ Loaded existing vector store from {CHROMA_DIR}.")
#         except Exception as e:
#             # Fallback if loading fails, force a re-index on next run
#             print(f"❌ Error loading existing vector store: {e}")
#             print("⚠️ Please delete the 'chroma_db' folder and run again to re-index.")
#             return

#     # 4. Run the Question Answering part
#     qa = make_retrieval_qa(vectordb)

#     print("\n✅ RAG Pipeline ready! Ask questions (type 'exit' to quit):")
#     while True:
#         q = input("\nQuestion: ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
        
#         # Note: RetrievalQA.from_chain_type returns a chain that uses the 'query' key in invoke.
#         # LangChain's RetrievalQA.from_chain_type creates a chain where the input variable for the
#         # user's question is named 'query' by default (unless otherwise specified in the prompt/chain config).
#         # We need to use the key 'query' here for the correct input to the chain.
#         # Alternatively, using run(q) is simpler for single-input chains.
#         try:
#              # Use invoke with 'query' key for LangChain 1.x style chain execution
#              result = qa.invoke({"query": q})
#              # The result from RetrievalQA.from_chain_type using stuff chain is a dict
#              # with 'result' key for the final answer.
#              print("\nAnswer:", result['result'])
#         except Exception as e:
#             print(f"An error occurred during QA: {e}")
#             break

# if __name__ == "__main__":
#     main()
# new code 

import os
from pathlib import Path
from dotenv import load_dotenv
import sys
from typing import Optional
import time

# --- LangChain and RAG Imports ---
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# --- Configuration & Global Variables ---
load_dotenv()
DATA_DIR = Path("data")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_4Lzxs9_GeePRj7CRLTJm2u8abEkqfR691awvjG2reqHvP9iYiSAphptZKJG2VcQfKSBiDF")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rishabh-portfolio")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# CRITICAL FIX: Set environment variable so langchain-pinecone can access it
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Google API Key
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDVhVk8IuSARft3CRZe7i-hymM7ttIF0lc")

# Global variables
HF_EMBEDDING_FUNCTION: Optional[HuggingFaceEmbeddings] = None
PINECONE_CLIENT: Optional[Pinecone] = None

# --- Core RAG Functions ---

def load_text_files(data_dir: Path):
    """Load text files."""
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
    """Load PDFs."""
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
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
                text = "\n".join(text_parts)
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
                del text_parts
            except Exception as e:
                print(f"⚠️ Could not read PDF {p.name}: {e}", file=sys.stderr)
                continue
    return docs

def ingest_documents(data_dir: Path):
    """Ingest documents."""
    print(f"📂 Loading documents from {data_dir}...")
    docs = load_text_files(data_dir) + load_pdfs(data_dir)
    print(f"✅ Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks."""
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

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists."""
    global PINECONE_CLIENT
    
    print(f"🔌 Connecting to Pinecone...")
    
    try:
        # Initialize Pinecone client
        PINECONE_CLIENT = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [index.name for index in PINECONE_CLIENT.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"📊 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            
            # Create index with serverless spec (free tier)
            PINECONE_CLIENT.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(10)
            print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully!")
        else:
            print(f"✅ Found existing index: {PINECONE_INDEX_NAME}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}", file=sys.stderr)
        return False

def create_or_load_vectorstore():
    """Create or load Pinecone vector store."""
    global HF_EMBEDDING_FUNCTION, PINECONE_CLIENT
    
    if not HF_EMBEDDING_FUNCTION:
        raise RuntimeError("Embedding function not initialized.")
    
    if not PINECONE_CLIENT:
        raise RuntimeError("Pinecone client not initialized.")
    
    try:
        # Get the index
        index = PINECONE_CLIENT.Index(PINECONE_INDEX_NAME)
        
        # Check if index has vectors
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        print(f"📊 Index stats: {vector_count} vectors")
        
        if vector_count == 0:
            print("📤 Index is empty. Uploading documents...")
            
            # Load and process documents
            docs = ingest_documents(DATA_DIR)
            if not docs:
                print("⚠️ No documents to index.")
                return None
            
            chunks = chunk_documents(docs)
            
            # Create vectorstore and upload to Pinecone
            # Environment variable is already set above, so no need to pass it here
            print(f"⬆️ Uploading {len(chunks)} chunks to Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=HF_EMBEDDING_FUNCTION,
                index_name=PINECONE_INDEX_NAME
            )
            print("✅ Documents uploaded to Pinecone!")
            
        else:
            print("📥 Loading existing vectors from Pinecone...")
            # Connect to existing vectorstore
            vectorstore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=HF_EMBEDDING_FUNCTION
            )
            print("✅ Connected to existing Pinecone vectorstore!")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error with Pinecone vectorstore: {e}", file=sys.stderr)
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return None

def make_retrieval_qa(vectorstore):
    """Create the QA chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
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
    """Initialize the complete RAG chain with Pinecone."""
    global HF_EMBEDDING_FUNCTION

    print("=" * 60)
    print("🚀 Initializing RAG Pipeline (Pinecone Cloud)")
    print("=" * 60)

    # 1. Load Embedding Model
    print(f"⌛ Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        HF_EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model loaded.")
    except Exception as e:
        print(f"❌ FATAL: Could not load embedding model. Error: {e}", file=sys.stderr)
        return None

    # 2. Initialize Pinecone
    if not initialize_pinecone():
        print("❌ FATAL: Could not initialize Pinecone.", file=sys.stderr)
        return None

    # 3. Create or Load Vectorstore
    vectorstore = create_or_load_vectorstore()
    if not vectorstore:
        print("❌ FATAL: Could not create/load vectorstore.", file=sys.stderr)
        return None

    # 4. Create QA Chain
    print("🔗 Creating QA Chain...")
    qa_chain = make_retrieval_qa(vectorstore)
    print("✅ QA Chain ready!")

    print("=" * 60)
    print("✅ RAG Pipeline Initialization Complete")
    print("=" * 60)

    return qa_chain

# Initialize the chain
QA_CHAIN = initialize_rag_chain()
