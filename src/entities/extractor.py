"""Lightweight deterministic entity extraction for support messages.

Order IDs, money amounts and dates often follow recognizable formats. Regex is transparent
and sufficient for these fields; a large NER model would add complexity without clear value.
"""
from __future__ import annotations
import re
from typing import Dict

ORDER_RE=re.compile(r"\border\s*(?:(?:id|number)\s*)?(?:is\s*)?(?:#|:|-)?\s*([A-Z0-9][A-Z0-9-]{3,})\b",re.I)
AMOUNT_RE=re.compile(r"((?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)\s*[\d,]+(?:\.\d{1,2})?)",re.I)
DATE_RE=re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?)\b",re.I)
EMAIL_RE=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

def extract_entities(text:str)->Dict:
    """Extract common support entities and return lists to handle multiple matches."""
    return {"order_ids":ORDER_RE.findall(text),"amounts":AMOUNT_RE.findall(text),
            "dates":DATE_RE.findall(text),"emails":EMAIL_RE.findall(text)}
