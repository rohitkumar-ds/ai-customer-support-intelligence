def test_classifier_prediction(tiny_classifier):
    result=tiny_classifier.predict_one("I need my money back")
    assert result["intent"]=="cancellation_and_refund" and 0<=result["confidence"]<=1
