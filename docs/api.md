# API

Start with:

```bash
uvicorn src.api.main:app --reload
```

Endpoints:
- `GET /health`
- `POST /analyze`
- `POST /chat`
- `POST /retrieve`
- `POST /predict-intent`
- `POST /knowledge/ingest`
- `GET /tickets`
- `GET /tickets/{ticket_id}`

`/analyze` performs classification/triage. `/chat` adds retrieval, grounded generation,
escalation and persistence. API keys are read from environment variables only.
