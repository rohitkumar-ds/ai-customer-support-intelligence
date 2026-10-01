"""Persistent FAISS vector store.

Code flow:
chunks -> Sentence Transformer embeddings -> normalized FAISS inner-product index
-> persisted index + JSON metadata.

"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

class FAISSVectorStore:
    """Build, persist, load, and search a cosine-similarity FAISS index."""

    def __init__(self,index_path:str,metadata_path:str):
        self.index_path=Path(index_path); self.metadata_path=Path(metadata_path)
        self.index=None; self.metadata=[]

    def build(self,chunks:List[Dict],embedder) -> None:
        """Embed chunks, build an IndexFlatIP index, and persist both artifacts."""
        if not chunks: raise ValueError("Cannot build a vector index from zero chunks.")
        try:
            import faiss
        except ImportError as exc: raise RuntimeError("Install faiss-cpu to build/search the vector index.") from exc
        vectors=embedder.encode([c["text"] for c in chunks]).astype("float32")
        faiss.normalize_L2(vectors)
        self.index=faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata=chunks
        self._save()

    def _save(self):
        self.index_path.parent.mkdir(parents=True,exist_ok=True)
        import faiss
        faiss.write_index(self.index,str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata,indent=2,ensure_ascii=False),encoding="utf8")

    def load(self):
        """Load an existing index and its metadata."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("FAISS index or metadata is missing. Run build_index.py.")
        import faiss
        self.index=faiss.read_index(str(self.index_path))
        self.metadata=json.loads(self.metadata_path.read_text(encoding="utf8"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index/vector metadata count mismatch.")
        return self

    def search(self,query:str,embedder,top_k:int=3,threshold:float=0.55)->List[Dict]:
        """Embed a query and return only chunks meeting the similarity threshold."""
        if not query.strip(): return []
        if self.index is None: raise RuntimeError("Vector store is not loaded.")
        import faiss
        vector=embedder.encode_one(query).astype("float32").reshape(1,-1)
        faiss.normalize_L2(vector)
        scores,indices=self.index.search(vector,min(top_k,self.index.ntotal))
        results=[]
        for score,idx in zip(scores[0],indices[0]):
            if idx < 0: continue
            if float(score) >= threshold:
                item=dict(self.metadata[int(idx)])
                item["similarity"]=round(float(score),4)
                results.append(item)
        return results
