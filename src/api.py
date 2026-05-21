from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import tempfile, shutil
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── App setup ─────────────────────────────────────────────────────────
# JD keywords: REST API, SDK integration, deployment, observability
app = FastAPI(
    title="Smart Document Intelligence API",
    description="RAG-powered document QA with NLP classification",
    version="1.0.0"
)
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# ── Global state ──────────────────────────────────────────────────────
retriever      = None
classifier     = None
startup_time   = None
request_count  = 0  # observability: track total requests


# ── Request/Response schemas ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class ClassifyRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    question:       str
    answer:         str
    sources:        list
    retrieval_time: float
    model_version:  str = "1.0.0"

class ClassifyResponse(BaseModel):
    text:           str
    predicted_class: str
    confidence:     float
    inference_time: float
    model_version:  str = "1.0.0"

class HealthResponse(BaseModel):
    status:        str
    uptime_seconds: float
    requests_served: int
    rag_loaded:    bool
    classifier_loaded: bool
    version:       str = "1.0.0"


# ── Startup: load models once ─────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    """
    Load RAG retriever and classifier at startup.
    JD keywords: model deployment, scalable AI systems, versioning
    """
    global retriever, classifier, startup_time
    startup_time = time.time()
    print("Loading models at startup...")

    # Load RAG retriever
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        if os.path.exists(CHROMA_DIR):
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            print("RAG retriever loaded.")
        else:
            print(f"Warning: ChromaDB not found at {CHROMA_DIR}")
            print("Run src/rag.py first to build the vector store.")
    except Exception as e:
        print(f"RAG load error: {e}")

    # Load document classifier
    try:
        from transformers import pipeline as hf_pipeline
        import json

        label_map_path = "data/processed/label_map.json"
        model_path     = "models/distilbert-doc-classifier/final"

        if os.path.exists(model_path) and os.path.exists(label_map_path):
            classifier = hf_pipeline(
                "text-classification",
                model=model_path,
                device=-1,          # CPU
                truncation=True,
                max_length=256
            )
            print("Document classifier loaded.")
        else:
            print("Warning: Classifier model not found.")
            print("Run src/train.py first to train the model.")
    except Exception as e:
        print(f"Classifier load error: {e}")

    print("API startup complete.")


# ── Middleware: request logging (observability) ───────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every request — JD keywords: observability, model monitoring
    """
    global request_count
    request_count += 1
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)
    print(f"[{request.method}] {request.url.path} "
          f"→ {response.status_code} ({duration}s)")
    return response


# ── Endpoint 1: Health check ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health/readiness endpoint.
    JD keywords: observability, model monitoring, deployment
    """
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - startup_time, 1),
        requests_served=request_count,
        rag_loaded=retriever is not None,
        classifier_loaded=classifier is not None
    )


# ── Endpoint 2: Document QA via RAG ──────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Answer questions using RAG retrieval.
    JD keywords: LLMs, RAG, information extraction, REST API,
                 responsible AI (sources returned for transparency)
    """
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="RAG retriever not loaded. Run src/rag.py first."
        )

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start = time.time()

    # Retrieve relevant chunks
    docs = retriever.invoke(req.question)

    # Build answer from most relevant chunk
    answer = docs[0].page_content.strip() if docs else "No relevant context found."

    retrieval_time = round(time.time() - start, 3)

    # Return sources for transparency (responsible AI)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "excerpt": doc.page_content[:120]
        }
        for doc in docs
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        retrieval_time=retrieval_time
    )


# ── Endpoint 3: Document classification ──────────────────────────────
@app.post("/classify", response_model=ClassifyResponse)
async def classify_document(req: ClassifyRequest):
    """
    Classify document text using fine-tuned DistilBERT.
    JD keywords: NLP, model deployment, inference, REST API
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not loaded. Run src/train.py first."
        )

    if not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    start = time.time()
    result = classifier(req.text[:512])[0]
    inference_time = round(time.time() - start, 3)

    return ClassifyResponse(
        text=req.text[:100] + "..." if len(req.text) > 100 else req.text,
        predicted_class=result["label"],
        confidence=round(result["score"], 4),
        inference_time=inference_time
    )


# ── Endpoint 4: Model info (versioning) ──────────────────────────────
@app.get("/model-info")
async def model_info():
    """
    Return model metadata.
    JD keywords: model versioning, observability
    """
    return {
        "api_version":        "1.0.0",
        "classifier_model":   "distilbert-base-uncased (fine-tuned)",
        "embedding_model":    EMBED_MODEL,
        "supported_classes":  [
            "comp.graphics",
            "rec.sport.hockey",
            "sci.med",
            "sci.space"
        ],
        "rag_vector_store":   "ChromaDB",
        "deployment_type":    "CPU inference",
    }

# ── Endpoint 5: File upload → add to RAG ─────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file and add it to the RAG vector store.
    JD keywords: data pipelines, data ingestion, scalable AI systems,
                 end-to-end ML lifecycle, dynamic data management
    """
    global retriever

    # Validate file type
    allowed = [".pdf", ".txt", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use PDF, TXT, or MD."
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Load document
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        # Add filename as metadata
        for doc in docs:
            doc.metadata["source"] = file.filename

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        # Embed and add to existing ChromaDB
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)

        # Reload retriever with updated store
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Clean up temp file
        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "chunks_added": len(chunks),
            "pages": len(docs),
            "status": "success",
            "message": f"Added {len(chunks)} chunks from '{file.filename}' to knowledge base. You can now ask questions about it."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ── Run server ────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )