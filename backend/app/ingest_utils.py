# backend/app/ingest_utils.py
import os
from pypdf import PdfReader
from typing import List
from haystack import Document

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages.append(txt)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunker with overlap (works for many doc types)."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)  # overlap
    return chunks

def docs_from_text_chunks(chunks: List[str], source: str) -> List[Document]:
    docs = []
    for i, c in enumerate(chunks):
        docs.append(Document(content=c, meta={"source": source, "chunk": i}))
    return docs
