import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = "data/chroma_db"
DOCS_DIR   = "data/docs"

def rebuild_chroma():
    print("Rebuilding ChromaDB vector store inside container...")
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs   = loader.load()
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"ChromaDB rebuilt at {CHROMA_DIR}")

if __name__ == "__main__":
    # Rebuild ChromaDB if it doesn't exist or is empty
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        rebuild_chroma()
    else:
        print("ChromaDB already exists, skipping rebuild.")

    # Start the API
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)