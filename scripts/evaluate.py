from pathlib import Path
import json
from src.config import get_settings
from src.evaluation.metrics import retrieval_metrics,save_json

RAG_CASES=[
    {"query":"How long before a refund appears on my card?","relevant":["faqs.jsonl","cancellation_and_refunds.md"]},
    {"query":"My tracking says delivered but the package is missing.","relevant":["faqs.jsonl","shipping_and_delivery.md"]},
    {"query":"My 2FA code does not arrive.","relevant":["faqs.jsonl","account_security.md"]},
]

def evaluate_rag():
    """Evaluate retrieval against a small manually curated source-ID test set."""
    from src.retrieval.retriever import Retriever
    r=Retriever()
    ranked=[]
    for case in RAG_CASES:
        ranked.append([x["document"] for x in r.search(case["query"],top_k=10,threshold=-1)])
    result=retrieval_metrics([c["relevant"] for c in RAG_CASES],ranked,3)
    save_json({"cases":RAG_CASES,"metrics":result,"note":"Small manual evaluation set; expand for reliable conclusions."},
              str(Path(get_settings().path("artifacts/evaluation_reports/rag_retrieval_report.json"))))
    return result

if __name__=="__main__":
    s=get_settings()
    if not Path(s.path(s.rag.index_path)).exists():
        print("RAG index is missing. Run: python scripts/build_index.py")
    else:
        print(evaluate_rag())
