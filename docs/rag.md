# RAG

## Ingestion

```text
Documents → extraction → chunking → metadata → embeddings → persistent FAISS index
```

TXT/Markdown files are loaded directly; PDFs are extracted page-by-page with PyMuPDF.
Chunks retain document/page metadata so the frontend can show citations.

## Query

```text
Question → embedding → FAISS search → top-K chunks → context → LLM → grounded answer
```

Retrieved documents are explicitly labeled as reference data, not instructions. This
helps resist prompt injection in documents. RAG reduces the need for the LLM to memorize
company policy, but it does not eliminate hallucinations.

Important parameters:
- `chunk_size`: maximum words in one chunk.
- `chunk_overlap`: repeated words between neighboring chunks.
- `top_k`: number of candidates retrieved.
- `similarity_threshold`: minimum cosine similarity accepted.
