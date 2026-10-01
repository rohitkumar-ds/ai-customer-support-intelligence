from __future__ import annotations
from src.config import get_settings
from src.embeddings.embedder import SentenceEmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

class Retriever:
    """Combine an embedding model and FAISS index behind one search method."""

    def __init__(self, embedder=None, store=None):
        settings=get_settings()
        self.embedder=embedder or SentenceEmbeddingModel(settings.rag.embedding_model)
        self.store=store or FAISSVectorStore(str(settings.path(settings.rag.index_path)),
                                             str(settings.path(settings.rag.metadata_path)))

    def search(self, query:str, top_k=None, threshold=None):
        """Return relevant chunks or an empty list when retrieval confidence is too low."""
        settings=get_settings()
        if self.store.index is None: self.store.load()
        return self.store.search(query,self.embedder,top_k or settings.rag.top_k,
                                 settings.rag.similarity_threshold if threshold is None else threshold)
