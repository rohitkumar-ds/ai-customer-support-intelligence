from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict

class LLMClient(ABC):
    """Minimal interface that keeps providers replaceable."""

    @abstractmethod
    def generate(self, query: str, context: str, history: List[Dict] | None = None) -> str:
        """Generate an answer from customer input and retrieved reference context."""
        raise NotImplementedError
