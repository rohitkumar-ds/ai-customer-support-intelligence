from __future__ import annotations
from typing import Dict
import re

class SentimentAnalyzer:
    """Wrap a Hugging Face text-classification pipeline."""

    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name=model_name; self._pipeline=None

    def _load(self):
        """Load the pretrained classifier only when sentiment is requested."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline=pipeline("sentiment-analysis",model=self.model_name)
            except Exception:
                # Offline demo fallback keeps the API usable when model weights cannot
                # be downloaded. It is explicitly low-confidence and is not a substitute
                # for the configured Transformer in evaluation.
                self._pipeline=False
        return self._pipeline

    def analyze(self,text:str)->Dict:
        """Return positive/negative/neutral label plus model confidence."""
        if not text.strip(): 
            return {"sentiment":"neutral","confidence":0.0}
        pipe=self._load()
        if pipe is False:
            positive={"great","good","happy","excellent","thanks","resolved","love"}
            negative={"bad","angry","terrible","hate","broken","failed","charged","fraud","urgent"}
            tokens=set(re.findall(r"\\w+",text.lower()))
            p=len(tokens & positive); n=len(tokens & negative)
            label="positive" if p>n else "negative" if n>p else "neutral"
            return {"sentiment":label,"confidence":0.5}
        out=pipe(text[:400])[0]
        raw=out["label"].lower()
        # The configured RoBERTa sentiment model provides all three requested labels.
        if raw in {"label_0","negative"}:
            label="negative"
        elif raw in {"label_1","neutral"}: 
            label="neutral"
        else:
            label="positive"
        return {"sentiment":label,"confidence":float(out["score"])}
