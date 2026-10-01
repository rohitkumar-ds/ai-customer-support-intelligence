from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

def classification_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    """Calculate accuracy, macro/weighted metrics, per-class metrics and confusion matrix."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true,y_pred)),4),
        "precision_macro": round(float(precision_score(y_true,y_pred,labels=labels,average="macro",zero_division=0)),4),
        "recall_macro": round(float(recall_score(y_true,y_pred,labels=labels,average="macro",zero_division=0)),4),
        "macro_f1": round(float(f1_score(y_true,y_pred,labels=labels,average="macro",zero_division=0)),4),
        "weighted_f1": round(float(f1_score(y_true,y_pred,labels=labels,average="weighted",zero_division=0)),4),
        "per_class": {k:{m:round(float(v),4) for m,v in val.items() if m in {"precision","recall","f1-score","support"}} for k,val in report.items() if k in labels},
        "confusion_matrix": confusion_matrix(y_true,y_pred,labels=labels).tolist(),
        "labels": labels,
    }

def evaluate_classifier(model, X_test: List[str], y_test: List[str], latency_iterations: int=30) -> Dict[str,Any]:
    """Evaluate a classifier and empirically benchmark single-query inference latency."""
    y_pred = model.predict(X_test)
    result = {"model_metadata": model.get_metadata(),
              "total_test_samples": len(X_test),
              "global_metrics": classification_metrics(y_test,y_pred,model.classes)}
    if X_test:
        model.predict([X_test[0]])
        times=[]
        for _ in range(latency_iterations):
            start=time.perf_counter(); model.predict([X_test[0]])
            times.append((time.perf_counter()-start)*1000)
        result["latency_ms"]={"mean":round(float(np.mean(times)),3),"p50":round(float(np.percentile(times,50)),3),
                              "p95":round(float(np.percentile(times,95)),3)}
    return result

def retrieval_metrics(relevant_ids: Sequence[Sequence[str]], ranked_ids: Sequence[Sequence[str]], k: int) -> Dict[str,float]:
    
    recalls = []
    precisions = []
    reciprocal = []

    for relevant, ranked in zip(relevant_ids, ranked_ids):
        rel = set(relevant)
        top = list(ranked[:k])

        # Count unique relevant sources only once.
        unique_top = set(top)
        hits = len(unique_top.intersection(rel))

        recalls.append(hits / len(rel) if rel else 0.0)

        # Precision is based on the number of unique retrieved sources.
        retrieved_count = len(unique_top)
        precisions.append(
            hits / retrieved_count if retrieved_count else 0.0
        )

        reciprocal.append(
            next(
                (1 / (i + 1) for i, x in enumerate(ranked) if x in rel),
                0.0
            )
        )

    return {
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "mrr": round(float(np.mean(reciprocal)), 4),
    }

def save_json(data: Dict, path: str) -> None:
    """Persist an evaluation dictionary as JSON."""
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2),encoding="utf8")

def save_evaluation_report(evaluation: Dict[str,Any], json_path: str, md_path: str) -> None:
    """Save JSON plus a readable Markdown representation."""
    save_json(evaluation,json_path)
    Path(md_path).parent.mkdir(parents=True,exist_ok=True)
    Path(md_path).write_text("# Evaluation Report\n\n```json\n"+json.dumps(evaluation,indent=2)+"\n```\n",encoding="utf8")
