import pytest
from src.models.classical import ClassicalIntentClassifier
@pytest.fixture
def tiny_classifier():
    m=ClassicalIntentClassifier()
    m.fit(["refund my payment","need money back","track my order","where is package","app crashes","server error","forgot password","cannot login"],
          ["cancellation_and_refund","cancellation_and_refund","order_tracking","order_tracking","technical_support","technical_support","account_access","account_access"])
    return m
