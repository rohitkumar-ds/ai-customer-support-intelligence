"""Practical grounding and escalation checks.
LLM confidence is not treated as a probability. The verifier combines retrieval evidence,
model intent confidence, and explicit business-risk rules to produce a qualitative confidence.
"""
from typing import Dict, List

def assess_confidence(intent_confidence: float, chunks: List[Dict], threshold: float) :
    """Return high/medium/low from observable support signals."""
    top=chunks[0]["similarity"] if chunks else 0.0
    if intent_confidence >= 0.8 and top >= threshold: return "high"
    if intent_confidence >= 0.6 and top >= threshold: return "medium"
    return "low"

def should_escalate(intent_confidence: float, sentiment: str, intent: str, chunks: List[Dict], settings) :
    """Apply deterministic human-escalation rules for uncertain or risky cases."""
    top=chunks[0]["similarity"] if chunks else 0.0
    if intent_confidence < settings.intelligence.confidence_threshold:
        return True,"Intent confidence is below the configured threshold."
    if top < settings.intelligence.escalation_similarity_threshold:
        return True,"Relevant company knowledge could not be confidently retrieved."
    if sentiment=="negative" and intent in settings.intelligence.critical_intents:
        return True,"High-risk support intent combined with negative sentiment."
    return False,""
