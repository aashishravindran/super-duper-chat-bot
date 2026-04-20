"""
Run this once to build the ChromaDB vector store and BM25 index from your resume PDF.

Usage:
    python scripts/ingest_resume.py
    python scripts/ingest_resume.py --pdf path/to/resume.pdf
"""
import argparse
from pathlib import Path

from src.rag.ingest import ingest

DEFAULT_PDF = Path("data/resume.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf}\nPlace your resume at {DEFAULT_PDF} or pass --pdf <path>")

    ingest(str(pdf))
    print("Done. Indexes saved to data/chroma_db/ and data/bm25_index.pkl")
