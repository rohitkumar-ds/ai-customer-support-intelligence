"""Configurable document chunking with metadata preservation.

Chunks are kept moderate in size so retrieval can return focused policy passages while
overlap preserves context that crosses a chunk boundary.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import List, Dict
from src.knowledge.loader import Document

class DocumentChunker:
    """Split documents into overlapping word chunks."""

    def __init__(self, chunk_size: int=800, chunk_overlap: int=120):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
        self.chunk_size,self.chunk_overlap=chunk_size,chunk_overlap

    def chunk(self, documents: List[Document]) -> List[Dict]:
        """Return chunk dictionaries with stable IDs and source metadata."""
        chunks=[]
        step=self.chunk_size-self.chunk_overlap
        for doc in documents:
            words=doc.text.split()
            for start in range(0,len(words),step):
                text=" ".join(words[start:start+self.chunk_size]).strip()
                if not text: continue
                base=doc.metadata.get("document","document")
                idx=len([c for c in chunks if c["document"]==base])
                chunks.append({"chunk_id":f"{PathSafe(base)}_{idx:04d}",
                               "text":text,"document":base,
                               "page":doc.metadata.get("page"),"section":None})
                if start+self.chunk_size >= len(words): break
        return chunks

def PathSafe(name: str) -> str:
    """Create a stable filesystem-safe identifier for a document name."""
    import re
    return re.sub(r"[^a-zA-Z0-9]+","_",name).strip("_").lower()
