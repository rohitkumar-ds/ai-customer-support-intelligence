"""TF-IDF + Logistic Regression intent classifier.

Code flow:
customer text -> classical preprocessing -> TF-IDF features -> Logistic Regression
-> intent + probability.

"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.models.base import BaseClassifier
from src.nlp.preprocessor import TextPreprocessor

class ClassicalIntentClassifier(BaseClassifier):
    """Trainable and serializable TF-IDF intent classifier."""

    def __init__(self, vectorizer=None, classifier=None, preprocessor=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.preprocessor = preprocessor or TextPreprocessor()

    def fit(self, texts: List[str], labels: List[str], max_features=10000,
            ngram_range=(1,2), min_df=1, max_df=0.98, C=1.0) -> "ClassicalIntentClassifier":
        """Fit TF-IDF and Logistic Regression on labeled customer messages."""
        cleaned = [self.preprocessor.preprocess_for_classical_ml(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df
        )
        X = self.vectorizer.fit_transform(cleaned)
        # Class weighting reduces the chance that common support intents hide rare classes.
        self.classifier = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42)
        self.classifier.fit(X, labels)
        return self

    @property
    def classes(self) -> List[str]:
        """Return intent labels learned during training."""
        return list(self.classifier.classes_) if self.classifier is not None else []

    def predict(self, texts: List[str]) -> List[str]:
        """Predict intent labels for one or more customer messages."""
        if self.classifier is None: 
            raise RuntimeError("Classical model is not fitted.")
        X = self.vectorizer.transform([self.preprocessor.preprocess_for_classical_ml(t) for t in texts])
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return class probabilities when Logistic Regression is used."""
        if self.classifier is None: 
            raise RuntimeError("Classical model is not fitted.")
        X = self.vectorizer.transform([self.preprocessor.preprocess_for_classical_ml(t) for t in texts])
        return self.classifier.predict_proba(X)

    def predict_one(self, text: str) -> Dict:
        """Return a convenient intent/confidence payload for an API caller."""
        probs = self.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return {"intent": self.classes[idx], "confidence": float(probs[idx])}

    def save(self, path: str) -> None:
        """Persist vectorizer, classifier and preprocessing settings with joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ClassicalIntentClassifier":
        """Load a previously trained classifier artifact."""
        return joblib.load(path)

    def get_metadata(self) -> Dict:
        """Describe the fitted model without exposing implementation internals."""
        return {
            "model_type": "tfidf_logistic_regression",
            "classes": self.classes,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "ngram_range": list(self.vectorizer.ngram_range) if self.vectorizer else [],
        }
