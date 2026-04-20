"""
PDF → chunks → ChromaDB + BM25 index.
Run once via scripts/ingest_resume.py; load at runtime via load_indexes().
"""
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from src.config import get_embeddings

CHROMA_DIR = Path("data/chroma_db")
BM25_PATH = Path("data/bm25_index.pkl")
COLLECTION = "resume"

# Section headers found in typical resumes — used for adaptive chunk sizing
_LARGE_SECTION_KEYWORDS = {"experience", "work", "project"}
_SMALL_SECTION_KEYWORDS = {"education", "skills", "summary", "contact"}


def _chunk_size_for_page(text: str) -> int:
    lower = text.lower()
    if any(k in lower for k in _LARGE_SECTION_KEYWORDS):
        return 512
    if any(k in lower for k in _SMALL_SECTION_KEYWORDS):
        return 256
    return 384


def ingest(pdf_path: str) -> tuple[Chroma, BM25Okapi, list[str]]:
    """Load PDF, chunk adaptively, embed into ChromaDB, build BM25 index."""
    pages = PyPDFLoader(pdf_path).load()

    docs = []
    for page in pages:
        size = _chunk_size_for_page(page.page_content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=64)
        docs.extend(splitter.split_documents([page]))

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

    # BM25 operates on simple token lists
    corpus = [doc.page_content.lower().split() for doc in docs]
    texts = [doc.page_content for doc in docs]
    bm25 = BM25Okapi(corpus)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts}, f)

    print(f"Ingested {len(docs)} chunks into ChromaDB and BM25 index.")
    return vectorstore, bm25, texts


def load_indexes() -> tuple[Chroma, BM25Okapi, list[str]]:
    """Load pre-built indexes from disk (called at runtime)."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return vectorstore, data["bm25"], data["texts"]
