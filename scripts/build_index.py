from src.config import get_settings
from src.knowledge.loader import KnowledgeLoader
from src.knowledge.chunker import DocumentChunker
from src.embeddings.embedder import SentenceEmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

def main():
    """Extract documents, chunk them, embed them and persist the FAISS index."""
    s=get_settings()
    docs=KnowledgeLoader(str(s.path(s.rag.knowledge_base_dir))).load()
    chunks=DocumentChunker(s.rag.chunk_size,s.rag.chunk_overlap).chunk(docs)
    embedder=SentenceEmbeddingModel(s.rag.embedding_model)
    store=FAISSVectorStore(str(s.path(s.rag.index_path)),str(s.path(s.rag.metadata_path)))
    store.build(chunks,embedder)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} source records.")

if __name__=="__main__": 
    main()
