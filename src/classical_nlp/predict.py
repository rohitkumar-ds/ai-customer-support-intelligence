"""Load the trained TF-IDF classifier and predict one message."""
import argparse
from src.config import get_settings
from src.models.classical import ClassicalIntentClassifier

def main():
    """Predict one configured message."""
    p=argparse.ArgumentParser(); p.add_argument("message")
    a=p.parse_args()
    model=ClassicalIntentClassifier.load(str(get_settings().path(get_settings().models.classical.artifact_path)))
    print(model.predict_one(a.message))

if __name__=="__main__": 
    main()
