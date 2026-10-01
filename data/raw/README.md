# Dataset notes

`tickets.jsonl` and `sample_tickets.jsonl` are **DEMO DATA — REPLACE WITH REAL DATASET**.
They are included only so the project has a small offline development fixture.

For a real public supervised benchmark, use:

```bash
python scripts/prepare_data.py --source banking77
```

The loader downloads Banking77 through Hugging Face `datasets` and normalizes it to the
project's `text` / `intent` interface.
