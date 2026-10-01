from src.embeddings.embedder import SentenceEmbeddingModel
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a=np.asarray(a); b=np.asarray(b)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denom) if denom else 0.0

def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and many vectors."""
    q=np.asarray(query); m=np.asarray(matrix)
    qn=np.linalg.norm(q); mn=np.linalg.norm(m,axis=1)
    return (m@q)/(np.maximum(mn,1e-12)*max(qn,1e-12))

BaseEmbedder=SentenceEmbeddingModel
SentenceTransformerEmbedder=SentenceEmbeddingModel
