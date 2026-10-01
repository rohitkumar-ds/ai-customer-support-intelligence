"""Train and evaluate the classical TF-IDF baseline.

Code flow:
dataset -> stratified splits -> preprocessing -> TF-IDF -> Logistic Regression
-> held-out evaluation -> model/report artifacts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from src.config import get_settings
from src.nlp.dataset import SupportDataset, save_splits
from src.models.classical import ClassicalIntentClassifier
from src.evaluation.metrics import evaluate_classifier, save_evaluation_report

def train(source=None):
    """Train the baseline and return the model plus genuine test metrics."""
    settings = get_settings()
    source = source or "local"
    ds = SupportDataset(source=source, path=str(settings.path("data/raw/tickets.jsonl")))
    df = ds.load()
    train_df, val_df, test_df = ds.split(df)
    save_splits(train_df, val_df, test_df)
    model = ClassicalIntentClassifier().fit(
        train_df.text.tolist(), train_df.intent.tolist(),
        max_features=settings.models.classical.max_features,
        ngram_range=tuple(settings.models.classical.ngram_range),
        min_df=settings.models.classical.min_df,
        max_df=settings.models.classical.max_df,
        C=settings.models.classical.c_regularization,
    )
    model.save(str(settings.path(settings.models.classical.artifact_path)))
    evaluation = evaluate_classifier(model, test_df.text.tolist(), test_df.intent.tolist())
    save_evaluation_report(evaluation, str(settings.path("artifacts/evaluation_reports/classical_model_report.json")),
                           str(settings.path("artifacts/evaluation_reports/classical_model_report.md")))
    return model, evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local","banking77"], default="local")
    args = parser.parse_args()
    model, metrics = train(args.source)
    print(metrics["global_metrics"])
