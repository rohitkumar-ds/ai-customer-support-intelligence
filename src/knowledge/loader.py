"""Load TXT, Markdown, and PDF support documents.

Code flow:
file discovery -> text extraction -> page/document metadata -> normalized document records.
PDF extraction uses PyMuPDF and retains page numbers for citations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class Document:
    text: str
    metadata: Dict[str, object]

class KnowledgeLoader:
    """Discover and extract supported knowledge-base files."""

    def __init__(self, directory: str):
        self.directory=Path(directory)

    def load(self) -> List[Document]:
        """Extract all supported files and return document/page records."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.directory}")
        docs=[]
        for path in sorted(self.directory.rglob("*")):
            if path.suffix.lower() in {".txt",".md"}:
                text=path.read_text(encoding="utf8")
                if text.strip(): docs.append(Document(text,{"document":path.name,"page":None,"path":str(path)}))
            elif path.suffix.lower()==".jsonl":
                import json

                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        line = line.strip()
                        if not line:
                            continue

                        record = json.loads(line)

                        question = str(record.get("question", "")).strip()
                        answer = str(record.get("answer", "")).strip()

                        if not question or not answer:
                            continue

                        text = f"Question: {question}\nAnswer: {answer}"

                        metadata = {
                            "document": path.name,
                            "page": None,
                            "path": str(path),
                            "faq_id": record.get("id"),
                            "category": record.get("category"),
                            "line": line_number,
                        }

                        docs.append(Document(text, metadata))

            elif path.suffix.lower()==".pdf":
                try:
                    import fitz
                except ImportError as exc:
                    raise RuntimeError("PyMuPDF is required for PDF ingestion.") from exc
                with fitz.open(path) as pdf:
                    for i,page in enumerate(pdf):
                        text=page.get_text("text")
                        if text.strip():
                            docs.append(Document(text,{"document":path.name,"page":i+1,"path":str(path)}))
        if not docs: raise ValueError("No TXT, Markdown, or PDF knowledge documents were found.")
        return docs
