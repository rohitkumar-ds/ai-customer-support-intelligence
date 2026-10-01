import json
from pathlib import Path
from src.embeddings.embedder import SentenceEmbeddingModel
import numpy as np

class FAQMatcher:
    """Search FAQ questions using sentence embeddings and cosine similarity."""

    def __init__(self, faq_file_path="data/knowledge_base/faqs.jsonl", embedder=None, similarity_threshold=0.55):
        self.path=Path(faq_file_path) 
        self.embedder=embedder or SentenceEmbeddingModel()
        self.threshold=similarity_threshold
        self.faqs=[json.loads(x) for x in self.path.read_text(encoding="utf8").splitlines() if x.strip()]
        self.embeddings=self.embedder.encode([f"{x['question']} {x.get('answer','')}" for x in self.faqs])

    def match(self,query,top_k=3):
        """Return ranked FAQ records with cosine similarity scores."""
        if not query.strip(): return []
        q=self.embedder.encode_one(query) 
        scores=self.embeddings @ q
        ids=np.argsort(scores)[::-1][:top_k]
        return [{**self.faqs[int(i)],"similarity":round(float(scores[i]),4),
                 "is_above_threshold":bool(scores[i]>=self.threshold),"rank":r+1} for r,i in enumerate(ids)]

    def get_most_similar_faq(self,query):
        """Return the best FAQ or None for an empty query."""
        results=self.match(query,1)
        return results[0] if results else None
