import json
from pathlib import Path
from src.config import get_settings

def main():
    """Print only metrics that actually exist in evaluation artifacts."""
    s=get_settings()
    for name in ["classical_model_report.json","transformer_model_report.json"]:
        p=s.path("artifacts/evaluation_reports")/name
        if p.exists():
            data=json.loads(p.read_text())
            metrics=data.get("global_metrics",data.get("global_metrics",{}))
            print(name,metrics)
        else:
            print(name,": NOT EVALUATED")

if __name__=="__main__": 
    main()
