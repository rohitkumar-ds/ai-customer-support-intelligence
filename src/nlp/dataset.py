"""Dataset loading and preparation for supervised intent classification.

The loader supports two sources:
1. A local JSONL file with ``text`` and ``intent`` fields.
2. The public Hugging Face ``banking77`` dataset, which is useful for a real,
   reproducible benchmark instead of claiming that demo records are a real dataset.

Code flow:
source -> records -> validation -> stratified train/validation/test split -> lists.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class SupportDataset:
    """Load customer-support intent examples from local JSONL or Hugging Face."""

    def __init__(self, source: str = "local", path: str = "data/raw/tickets.jsonl", seed: int = 42):
        self.source, self.path, self.seed = source, Path(path), seed

    def load(self) -> pd.DataFrame:
        """Load and validate records, raising clear errors for malformed datasets."""
        if self.source == "banking77":
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("Install the 'datasets' package to use banking77.") from exc
            ds = load_dataset("mteb/banking77")
            frames = []
            for split in ds:
                frame = ds[split].to_pandas()
                frame = frame.rename(columns={"text": "text", "label_text": "intent"})
                frames.append(frame[["text", "intent"]])
            return pd.concat(frames, ignore_index=True).dropna()

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        required = {"text", "intent"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        frame = frame[["text","intent"]].dropna()
        frame["text"] = frame["text"].astype(str)
        frame["intent"] = frame["intent"].astype(str)
        return frame.drop_duplicates(subset=["text"]).reset_index(drop=True)

    def split(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        """Create stratified train/validation/test splits where class counts permit it."""
        if df["intent"].nunique() < 2:
            raise ValueError("At least two intent classes are required.")
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=self.seed, stratify=df["intent"]
        )
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=relative_val, random_state=self.seed, stratify=train_val["intent"]
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def save_splits(train, val, test, output_dir: str = "data/processed") -> None:
    """Save standardized JSONL splits for reproducible training."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, frame in {"train":train,"val":val,"test":test}.items():
        frame.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
