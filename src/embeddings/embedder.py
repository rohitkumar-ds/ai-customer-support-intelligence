"""Semantic embedding wrapper.

Code flow:
text -> SentenceTransformer -> dense vector -> optional L2 normalization.

Unlike TF-IDF, embeddings place semantically related sentences near each other even
when they do not share the same words. The model is loaded lazily so importing the
application does not download a model.
"""
from __future__ import annotations
from typing import List
import numpy as np

class SentenceEmbeddingModel:
    """Lazy Sentence Transformer facade used by semantic search and RAG."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name=model_name
        self._model=None

    def _load(self):
        """Load the configured model only when an embedding is first requested."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for embeddings.") from exc
            self._model=SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the model's vector dimension after loading."""
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: List[str] | str) -> np.ndarray:
        """Encode one string or a list of strings as normalized float32 vectors."""
        values=[texts] if isinstance(texts,str) else texts
        if not values:
            return np.empty((0,0),dtype=np.float32)
        return np.asarray(self._load().encode(values, normalize_embeddings=True),dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode one query and return a one-dimensional vector."""
        return self.encode(text)[0]
