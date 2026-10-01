from __future__ import annotations
import re
from dataclasses import dataclass
from html import unescape


@dataclass
class PreprocessorConfig:
    """Serializable preprocessing configuration."""

    strip_html: bool = True
    lowercase: bool = True
    remove_special_chars: bool = False


class TextPreprocessor:
    """Preprocess customer text according to the selected NLP pipeline."""

    def __init__(self, config=None):
        # A normal module-level dataclass is used so the preprocessor
        # can be safely serialized with joblib.
        self.config = config or PreprocessorConfig()

    def normalize(self, text: str) -> str:
        """Normalize whitespace and HTML while preserving meaningful punctuation."""
        if text is None:
            return ""

        value = unescape(str(text))

        if self.config and self.config.strip_html:
            value = re.sub(r"<[^>]+>", " ", value)

        value = re.sub(r"\s+", " ", value).strip()
        return value

    def preprocess_for_classical_ml(self, text: str) -> str:
        """Prepare text for TF-IDF without destroying useful support terminology."""
        value = self.normalize(text)

        if self.config and self.config.lowercase:
            value = value.lower()

        if self.config and self.config.remove_special_chars:
            value = re.sub(r"[^a-z0-9₹$€£\s]", " ", value)

        return re.sub(r"\s+", " ", value).strip()

    def preprocess_for_transformer(self, text: str) -> str:
        """Prepare text for a Transformer while preserving natural word order and syntax."""
        # BERT/DistilBERT has its own subword tokenizer. Removing stopwords or
        # punctuation here can remove signals learned during pretraining.
        return self.normalize(text)

    def clean(self, text: str) -> str:
        """Backward-compatible alias for the classical preprocessing path."""
        return self.preprocess_for_classical_ml(text)
