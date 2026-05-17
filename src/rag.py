import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline

DOCS_DIR    = "data/docs"
CHROMA_DIR  = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    print("[1/5] Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"      Loaded {len(docs)} documents")
    for d in docs:
        print(f"      - {d.metadata['source']}  ({len(d.page_content)} chars)")
    return docs


def chunk_and_embed(docs):
    print("\n[2/5] Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"      Sample chunk: '{chunks[0].page_content[:100]}...'")

    print("\n      Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("      Embedding model loaded.")
    return chunks, embeddings


def build_vector_store(chunks, embeddings):
    print("\n[3/5] Building ChromaDB vector store...")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"      Indexed {len(chunks)} chunks -> persisted to {CHROMA_DIR}")
    return vectorstore


def build_rag_chain(vectorstore):
    print("\n[4/5] Building retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    print("\n[5/5] Loading LLM and building RAG chain...")
    generator = hf_pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2",   # tiny model, fast, CPU-friendly
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=50256,
        truncation=True,
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=generator)

    prompt = PromptTemplate.from_template(
        "Context: {context}\nQ: {question}\nA:"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("      RAG chain ready.")
    return rag_chain, retriever


def query_rag(rag_chain, retriever, question):
    print(f"\n{'─'*50}")
    print(f"Question: {question}")
    print(f"{'─'*50}")

    # Retrieve relevant chunks directly
    source_docs = retriever.invoke(question)

    # Show the most relevant chunk as the answer
    # (tiny-gpt2 cannot generate coherent answers — in production
    #  swap for GPT-4/Claude API in one line)
    best_chunk = source_docs[0].page_content.strip()
    print(f"Most relevant context retrieved:\n  {best_chunk[:200]}")
    print(f"\nSources ({len(source_docs)} chunks):")
    for i, doc in enumerate(source_docs):
        src = doc.metadata.get("source", "unknown")
        print(f"  [{i+1}] {src}")

    return best_chunk

if __name__ == "__main__":
    print("=" * 50)
    print("RAG Pipeline — Smart Document Intelligence")
    print("=" * 50)

    docs             = load_documents()
    chunks, emb      = chunk_and_embed(docs)
    vectorstore      = build_vector_store(chunks, emb)
    rag_chain, retriever = build_rag_chain(vectorstore)

    test_questions = [
        "What is supervised learning?",
        "What metrics are used to evaluate classification models?",
        "What is overfitting and how can it be prevented?",
        "What are Large Language Models used for?",
    ]

    print("\n" + "=" * 50)
    print("Running test queries...")
    print("=" * 50)

    for q in test_questions:
        query_rag(rag_chain, retriever, q)

    print("\n" + "=" * 50)
    print("RAG pipeline complete. Day 5-6 done.")
    print("=" * 50)