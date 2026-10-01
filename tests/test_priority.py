from src.intelligence.priority import determine_priority

def test_financial_issue_high(): 
    assert determine_priority("billing_and_payments","negative",.9,{"amounts":["₹100"]},"charged twice")=="high"

def test_simple_inquiry_low(): 
    assert determine_priority("general_inquiry","positive",.95,{},"what are your hours?")=="low"
