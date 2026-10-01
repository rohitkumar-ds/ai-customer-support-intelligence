import re
from typing import List

def simple_tokenize(text: str) -> List[str]:
    """Tokenize a string into lowercase word/number-like tokens for demonstrations."""
    return re.findall(r"\b[\w₹$€£'-]+\b", text.lower())
