from typing import Dict

def determine_priority(intent:str,sentiment:str,intent_confidence:float,entities:Dict, message:str)->str:
    """Return low/medium/high/critical priority using explicit support rules."""
    text=message.lower()
    if intent=="account_access" and any(x in text for x in ["suspicious","unknown ip","hacked","takeover"]):
        return "critical"
    if intent=="technical_support" and any(x in text for x in ["production down","outage","all users","critical"]):
        return "critical"
    if intent in {"billing_and_payments","cancellation_and_refund"} and entities.get("amounts"):
        return "high"
    if sentiment=="negative" or intent_confidence<0.60:
        return "high" if intent in {"billing_and_payments","account_access","technical_support"} else "medium"
    if intent in {"billing_and_payments","account_access","technical_support","cancellation_and_refund"}:
        return "medium"
    return "low"
