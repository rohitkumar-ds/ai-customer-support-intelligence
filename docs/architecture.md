# Architecture

```text
Customer
  ↓
Streamlit / FastAPI
  ↓
Query Analysis
  ├─ TF-IDF + Logistic Regression baseline
  ├─ DistilBERT intent classifier (trained separately)
  ├─ Transformer sentiment analyzer
  └─ Regex entities
  ↓
Priority + Department Decision Engine
  ↓
Query Embedding
  ↓
FAISS top-K retrieval + similarity threshold
  ↓
RAG context builder
  ↓
LLM provider abstraction
  ↓
Grounding verifier + escalation
  ↓
SQLite ticket/audit records
```

The key architectural boundary is that model outputs are evidence. Business priority,
routing and escalation are explicit deterministic decisions rather than instructions
delegated to the LLM.
