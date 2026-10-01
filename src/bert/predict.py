"""Predict an intent using a saved DistilBERT classifier."""
import argparse
from src.config import get_settings
from src.models.transformer import TransformerIntentClassifier
def main():
    """Load the configured checkpoint and classify a message."""
    p=argparse.ArgumentParser(); 
    p.add_argument("message"); 
    a=p.parse_args()
    s=get_settings() 
    model=TransformerIntentClassifier(s.models.transformer.model_name,s.models.transformer.max_length)
    model.load(str(s.path(s.models.transformer.artifact_path)))
    print(model.predict_one(a.message))

if __name__=="__main__":
     main()
