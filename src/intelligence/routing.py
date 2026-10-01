"""Configuration-driven intent-to-department routing."""
DEFAULT_ROUTING={
    "billing_and_payments":"billing",
    "cancellation_and_refund":"billing",
    "order_tracking":"logistics",
    "technical_support":"technical_support",
    "account_access":"account_support",
    "general_inquiry":"customer_support",
}
def route_department(intent:str, mapping=None)->str:
    """Map an intent to its operational department."""
    return (mapping or DEFAULT_ROUTING).get(intent,"customer_support")
