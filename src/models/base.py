"""Interfaces shared by intent classifiers.

The interface keeps API and orchestration code independent from the model family,
so TF-IDF and Transformer classifiers can be compared without changing callers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseClassifier(ABC):
    """Common prediction contract for intent models."""

    @property
    @abstractmethod
    def classes(self) -> List[str]: ...

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]: ...

    @abstractmethod
    def predict_proba(self, texts: List[str]): ...

    @abstractmethod
    def get_metadata(self) -> Dict: ...
