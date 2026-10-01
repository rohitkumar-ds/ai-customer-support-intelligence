"""DistilBERT intent classifier for the Transformer comparison.

Code flow:
raw text -> DistilBERT tokenizer -> input IDs + attention mask -> pretrained encoder
-> classification head -> intent probabilities.

"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

class TransformerIntentClassifier:
    """Hugging Face sequence classifier with save/load and single-query prediction."""

    def __init__(self, model_name="distilbert-base-uncased", max_length=128, label2id=None, id2label=None):
        self.model_name=model_name; self.max_length=max_length
        self.label2id=label2id or {}; self.id2label=id2label or {}
        self.tokenizer=None; self.model=None

    def load(self,path:str):
        """Load tokenizer and fine-tuned model from a training artifact directory."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc: 
            raise RuntimeError("transformers is required.") from exc
        self.tokenizer=AutoTokenizer.from_pretrained(path)
        self.model=AutoModelForSequenceClassification.from_pretrained(path)
        self.id2label={int(k):v for k,v in self.model.config.id2label.items()} if self.model.config.id2label else self.id2label
        self.label2id={v:k for k,v in self.id2label.items()}
        return self

    def predict_one(self,text:str)->Dict:
        """Tokenize one message and return predicted intent and softmax confidence."""
        if self.model is None or self.tokenizer is None: 
            raise RuntimeError("Transformer model is not loaded.")
        import torch
        self.model.eval()
        inputs=self.tokenizer(text,return_tensors="pt",truncation=True,max_length=self.max_length)
        with torch.no_grad(): 
            logits=self.model(**inputs).logits
        probs=torch.softmax(logits,dim=-1)[0].cpu().numpy()
        idx=int(np.argmax(probs))
        return {"intent":self.id2label.get(idx,str(idx)),"confidence":float(probs[idx])}

    def predict(self,texts:List[str])->List[str]:
        """Predict intents for multiple messages."""
        return [self.predict_one(t)["intent"] for t in texts]

    @property
    def classes(self)->List[str]:
        """Return model label names."""
        return [self.id2label[i] for i in sorted(self.id2label)]

    def get_metadata(self)->Dict:
        """Return model metadata for evaluation and comparison."""
        return {"model_type":"distilbert_intent_classifier","model_name":self.model_name,
                "classes":self.classes,"max_length":self.max_length}
