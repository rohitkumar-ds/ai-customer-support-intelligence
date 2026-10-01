# Embeddings and semantic search

TF-IDF is sparse and strongly tied to vocabulary overlap. Sentence embeddings represent
the meaning of a sentence in a dense vector space. This makes them useful when a customer
uses different words from the FAQ.

The configured model is `sentence-transformers/all-MiniLM-L6-v2`.

Normalized vectors allow FAISS inner product to represent cosine similarity. Top-K
retrieval limits the context to the strongest candidates. A similarity threshold prevents
weak matches from being passed to the generator.
