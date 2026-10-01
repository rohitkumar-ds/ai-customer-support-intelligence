# AI Customer Support Intelligence Platform

An end-to-end customer support intelligence platform combining **classical NLP, Transformer-based classification, semantic retrieval, Retrieval-Augmented Generation (RAG), LLM generation, deterministic ticket intelligence, FastAPI, Streamlit, SQLite and evaluation workflows**.

The project is designed to demonstrate how a customer-support request can move from **text analysis → intent classification → sentiment/entity extraction → priority and routing → knowledge retrieval → grounded response → escalation and audit storage**.

> **Important:** `data/raw/tickets.jsonl` is a small local demo dataset for development and offline testing. Its evaluation results must not be interpreted as Banking77 or production performance. The project also supports the public **Banking77** dataset for a larger supervised classification benchmark.

---

## Overview

A customer-support system should do more than generate a conversational response.

This project separates different responsibilities:

* **Intent classification** identifies the type of customer problem.
* **Sentiment analysis** identifies positive, neutral or negative sentiment.
* **Entity extraction** identifies operational information such as order IDs, amounts, dates and email addresses.
* **Deterministic decision rules** determine priority and department routing.
* **Sentence embeddings + FAISS** retrieve relevant support knowledge.
* **RAG + LLM** generates a response grounded in retrieved information.
* **Verification and escalation** prevent weak evidence from being presented as confident policy.
* **SQLite** stores tickets, triage information, sources, conversations and feedback.

The LLM is therefore not responsible for making critical business-priority decisions.

---

## Example Workflow

For a customer message such as:

```text
"My card was charged ₹4,999 twice for order A78291 and I still haven't received a refund."
```

the platform can produce information such as:

```text
Intent       → billing_and_payments
Sentiment    → negative
Entities     → order A78291, amount ₹4,999
Priority     → high
Department   → billing
Retrieval    → relevant payment/refund knowledge
RAG          → grounded response
Sources      → document/page/chunk metadata
Escalation   → deterministic decision when evidence is weak or risky
Storage      → SQLite ticket/audit records
```

Exact outputs depend on the models, configuration and evidence available at runtime.

---

## Architecture

```text
                         Customer
                            │
                            ▼
                    Streamlit / FastAPI
                            │
                            ▼
                       Query Analysis
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Intent       Sentiment      Entities
          TF-IDF /      Transformer      Regex
          DistilBERT
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   Priority + Routing
                            │
                            ▼
                    Semantic Retrieval
                            │
                            ▼
                       FAISS / RAG
                            │
                            ▼
                      LLM Provider
                            │
                            ▼
                  Grounded Response
                       │          │
                       ▼          ▼
                    Sources   Escalation
                       │          │
                       └────┬─────┘
                            ▼
                          SQLite
                            │
                            ▼
                        Analytics
```

### Core design principle

**Model outputs provide evidence; final business decisions are centralized and deterministic.**

The LLM does not independently determine critical ticket priority.

---

## NLP & AI Pipeline

### 1. Classical NLP

```text
Raw Text
   ↓
Classical preprocessing
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Intent
```

The classical baseline provides a lightweight and interpretable reference point before introducing Transformer-based classification.

The default TF-IDF configuration includes:

```yaml
max_features: 10000
ngram_range: [1, 2]
min_df: 1
max_df: 0.98
c_regularization: 1.0
```

---

### 2. Transformer Classification

```text
Raw Text
   ↓
DistilBERT tokenizer
   ↓
Input IDs + Attention Mask
   ↓
Pretrained DistilBERT
   ↓
Classification Head
   ↓
Intent
```

The project uses:

```text
distilbert-base-uncased
```

The model is fine-tuned rather than trained from scratch.

---

### 3. Sentiment Analysis

The sentiment component uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

and produces:

```text
positive
neutral
negative
```

with a model classification score.

The score should be treated as a model signal rather than an objective measurement of a customer's emotional state.

---

### 4. Entity Extraction

The project includes deterministic extraction for operational information such as:

* Order IDs
* Amounts
* Dates
* Email addresses

This information can then be used by downstream ticket intelligence and routing components.

---

## Retrieval-Augmented Generation

The RAG pipeline is:

```text
Customer Question
       ↓
Query Embedding
       ↓
FAISS Search
       ↓
Top-K Knowledge Chunks
       ↓
Context Builder
       ↓
Controlled Prompt
       ↓
LLM
       ↓
Grounded Response
```

The prompt separates:

```text
SYSTEM INSTRUCTIONS
RETRIEVED KNOWLEDGE
CUSTOMER INPUT
```

Retrieved documents are treated as reference information rather than instructions, providing a basic layer of prompt-injection awareness.

---

## Knowledge Ingestion

The system supports knowledge ingestion from:

* TXT
* Markdown
* PDF

Documents are extracted, chunked and enriched with metadata such as:

```text
document
page
section
chunk_id
```

Default chunk configuration:

```yaml
chunk_size: 800
chunk_overlap: 120
```

The vector store uses:

```text
Sentence Transformer
        ↓
Normalized vectors
        ↓
FAISS IndexFlatIP
```

Default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Because the vectors are normalized, inner product corresponds to cosine similarity.

Default retrieval configuration:

```yaml
top_k: 3
similarity_threshold: 0.55
```

If no retrieved chunk passes the similarity threshold, the system can return without forcing irrelevant evidence into the LLM context.

---

## Grounding & Escalation

The system includes several safeguards around generated responses:

1. Similarity thresholds for retrieval.
2. Rejection of weak or empty retrieval results.
3. Instructions to avoid inventing company policy.
4. Preservation of policy numbers and dates.
5. Source metadata in responses.
6. Qualitative confidence levels.
7. Deterministic escalation rules.

Example fallback behavior:

```text
I couldn't find sufficient information in the available
support knowledge. This case should be reviewed by a
support agent.
```

RAG can **reduce** hallucination risk, but it does not eliminate hallucinations.

---

## Confidence Framework

The project does not treat an LLM-generated probability as a reliable confidence score.

Instead, operational confidence combines signals such as:

```text
Intent confidence
        +
Retrieval similarity
        +
Source availability
        +
Escalation rules
        ↓
high / medium / low
```

This is an operational signal rather than statistical certainty.

---

## Human Escalation

Escalation can occur when:

* intent confidence is too low
* retrieval confidence is too low
* required company policy cannot be found
* a high-risk account/security issue is detected
* negative sentiment accompanies a critical support intent
* another deterministic rule requires human review

The LLM does not override these rules.

---

## LLM Provider Architecture

The LLM layer is isolated behind a provider interface:

```python
LLMClient.generate(...)
```

Current implementations include:

```text
MockLLMClient
GeminiLLMClient
```

The default configuration uses:

```yaml
llm:
  provider: mock
```

The mock provider allows tests and offline demonstrations without requiring a paid LLM API.

For live Gemini generation:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

**Never commit a real API key to GitHub.**

---

## Data

### Local demo dataset

```text
data/raw/tickets.jsonl
```

The included dataset is intentionally small and is primarily useful for:

* development
* smoke testing
* offline demonstrations
* verifying the end-to-end pipeline

### Banking77

The project includes a configurable loader for the public Hugging Face **Banking77** dataset.

Example:

```bash
python scripts/prepare_data.py --source banking77
python -m src.models.train_classical --source banking77
python -m src.models.train_transformer --source banking77
```

A local JSONL dataset with the following fields can also be used:

```text
text
intent
```

---

## Evaluation

### Classification

The project supports comparison between:

```text
TF-IDF + Logistic Regression
            VS
        DistilBERT
```

Evaluation includes:

* Accuracy
* Precision
* Recall
* Macro F1
* Weighted F1
* Per-class metrics
* Confusion matrix
* Inference latency

### Verified demo result

The included demo dataset was evaluated with:

```text
Test samples : 20
Accuracy     : 0.8000
Macro F1     : 0.7762
Weighted F1  : 0.7861
```

**Important:** these results are only for the included demo data. They are **not Banking77 results and are not production performance claims**.

Run the Banking77 training/evaluation workflow to obtain a meaningful public benchmark.

### Retrieval evaluation

The project supports:

* Recall@K
* Precision@K
* MRR

The included RAG evaluation set is intentionally small and manually curated, so it should be considered an educational evaluation rather than a production benchmark.

### Generation evaluation

Possible evaluation dimensions include:

* Answer relevance
* Context relevance
* Faithfulness

LLM-as-a-judge metrics should be treated as evaluation signals rather than ground truth.

---

## Testing

Run:

```bash
pytest -q
```

The test suite covers components including:

* preprocessing
* classifier prediction
* entity extraction
* priority engine
* chunking
* RAG context construction
* retrieval metrics
* API route registration

Tests are designed so that they do not require a paid LLM call.

---

## FastAPI

Start the API with:

```bash
uvicorn src.api.main:app --reload
```

or:

```bash
python run.py api
```

### Available endpoints

```text
GET  /health
POST /analyze
POST /chat
POST /retrieve
POST /predict-intent
POST /knowledge/ingest
GET  /tickets
GET  /tickets/{ticket_id}
```

### Example `/analyze`

Request:

```json
{
  "message": "My payment was deducted twice."
}
```

The response can include:

```text
intent
intent_confidence
sentiment
sentiment_confidence
entities
priority
department
latency_ms
```

### Example `/chat`

Request:

```json
{
  "message": "How long does a refund take?",
  "conversation_id": "conv_123"
}
```

The response can include:

```text
answer
intent
sentiment
priority
department
sources
confidence
escalate
escalation_reason
conversation_id
ticket_id
```

---

## Streamlit Dashboard

Start the dashboard with:

```bash
streamlit run dashboard/app.py
```

or:

```bash
python run.py dashboard
```

The dashboard includes:

1. Home
2. AI Support Chat
3. Ticket Analyzer
4. Knowledge Search
5. Ticket History
6. Analytics
7. Model Comparison
8. RAG Evaluation

The dashboard exposes intermediate triage signals and retrieval sources instead of hiding the complete process behind a single chatbot response.

---

## SQLite Storage

SQLite is used for local ticket and audit storage.

The system stores information across areas including:

```text
tickets
triage_records
rag_logs
sources
conversations
feedback
```

Ticket records can include:

```text
id
created_at
message
response
confidence
escalated
escalation_reason
```

Triage records can include:

```text
intent
intent_confidence
sentiment
sentiment_confidence
priority
department
model_name
inference_time_ms
```

The current implementation is designed for educational/local use.

---

## Project Structure

```text
ai-customer-support-intelligence/
│
├── src/
│   ├── nlp/
│   ├── models/
│   ├── embeddings/
│   ├── knowledge/
│   ├── retrieval/
│   ├── rag/
│   ├── llm/
│   ├── entities/
│   ├── intelligence/
│   ├── services/
│   ├── database/
│   └── api/
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── configs/
│   ├── config.yaml
│   ├── dataset.yaml
│   └── logging_config.yaml
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── evaluate.py
│   └── evaluate_models.py
│
├── docs/
│   ├── architecture.md
│   ├── nlp_pipeline.md
│   ├── bert.md
│   ├── embeddings.md
│   ├── rag.md
│   ├── evaluation.md
│   └── api.md
│
├── data/
├── artifacts/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── run.py
```

Detailed technical documentation is available in the `docs/` directory.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rohitkumar-ds/ai-customer-support-intelligence.git
cd ai-customer-support-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Then configure the variables required for your local environment.

For the default mock LLM configuration, a Gemini API key is not required.

---

## Build the Knowledge Index

After preparing the knowledge documents, build the vector index:

```bash
python scripts/build_index.py
```

The generated vector index and model artifacts are intentionally not committed to Git.

---

## Run the Application

### API

```bash
python run.py api
```

### Streamlit

```bash
python run.py dashboard
```

### Tests

```bash
pytest -q
```

---

## Key Technologies

| Area            | Technologies                  |
| --------------- | ----------------------------- |
| Programming     | Python                        |
| Classical NLP   | TF-IDF, Logistic Regression   |
| Transformer NLP | DistilBERT                    |
| Sentiment       | RoBERTa-based classifier      |
| Embeddings      | Sentence Transformers         |
| Vector Search   | FAISS                         |
| Generative AI   | RAG, LLM provider abstraction |
| LLM Provider    | Mock LLM, Gemini adapter      |
| API             | FastAPI                       |
| Dashboard       | Streamlit                     |
| Database        | SQLite                        |
| Testing         | Pytest                        |
| Configuration   | YAML, environment variables   |

---

## Design Decisions

### Why use both classical NLP and Transformers?

The classical model provides a lightweight baseline that is easy to train and evaluate. The Transformer model provides a stronger NLP architecture for comparison.

### Why use RAG?

Instead of expecting an LLM to know company-specific support policy, the system retrieves relevant knowledge and supplies it as context for generation.

### Why deterministic priority rules?

Business-critical routing decisions should be transparent and reproducible rather than being determined solely by generated text.

### Why a Mock LLM?

It allows the application and test suite to run without requiring a paid external LLM API.

---

## Limitations

This is an educational and portfolio project rather than a production customer-support system.

Current limitations include:

1. RAG does not completely eliminate hallucinations.
2. Retrieval quality depends on the embedding model and chunking strategy.
3. Transformer classification requires suitable labeled data.
4. LLM-generated output can still be incorrect.
5. Semantic similarity does not guarantee factual correctness.
6. Confidence is an operational signal rather than absolute truth.
7. The included demo dataset is too small for production claims.
8. Conversation memory is currently bounded and in-process.
9. The included RAG evaluation set is intentionally small.
10. Production deployment would require stronger security, observability, access control, privacy controls and human-review workflows.

---

## Future Improvements

Potential extensions include:

* Larger domain-specific intent datasets
* Probability calibration
* Cross-encoder retrieval reranking
* Stronger retrieval evaluation datasets
* Human feedback collection
* Persistent conversation storage
* PII redaction
* Authentication and role-based access
* Production observability
* Stronger answer-faithfulness evaluation
* Multilingual support

---

## Documentation

Detailed technical documentation is available in:

```text
docs/
├── architecture.md
├── nlp_pipeline.md
├── bert.md
├── embeddings.md
├── rag.md
├── evaluation.md
└── api.md
```

These documents provide deeper explanations of the individual components without making the main README unnecessarily long.

---

## What This Project Demonstrates

This project demonstrates an end-to-end approach to building an AI-powered support system:

```text
Data Preparation
      ↓
Classical NLP Baseline
      ↓
Transformer Classification
      ↓
Semantic Retrieval
      ↓
Knowledge Indexing
      ↓
RAG
      ↓
LLM Generation
      ↓
Verification
      ↓
Deterministic Priority / Routing
      ↓
Human Escalation
      ↓
API + Dashboard
      ↓
Storage + Analytics
      ↓
Testing + Evaluation
```

The focus is on **NLP, machine learning, retrieval, RAG, grounded generation, business logic and measurable evaluation**, rather than adding infrastructure complexity that is not required by the problem.

---

## License

This project is provided for educational and portfolio use. See [`LICENSE`](LICENSE) for the applicable license terms.

---

## Author

**Rohit Kumar**

Computer Science Engineering Student
Data Science | Machine Learning | AI

GitHub: [@rohitkumar-ds](https://github.com/rohitkumar-ds)
