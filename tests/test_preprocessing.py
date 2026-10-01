from src.nlp.preprocessor import TextPreprocessor

def test_classical_normalization():
     assert TextPreprocessor().preprocess_for_classical_ml("  Hello <b>WORLD</b>  ")=="hello world"

def test_transformer_keeps_natural_text(): 
     assert "twice" in TextPreprocessor().preprocess_for_transformer("Charged twice!")
