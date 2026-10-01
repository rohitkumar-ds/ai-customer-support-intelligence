from src.evaluation.metrics import retrieval_metrics

def test_metrics():
    out=retrieval_metrics([["a"]],[["a","b"]],2)
    assert out["recall_at_k"]==1 and out["mrr"]==1
