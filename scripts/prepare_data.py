import argparse
from src.config import get_settings
from src.nlp.dataset import SupportDataset, save_splits

def main():
    """Load the selected dataset and write reproducible train/val/test JSONL files."""
    p=argparse.ArgumentParser(); p.add_argument("--source",choices=["banking77","local"],default="banking77")
    args=p.parse_args()
    s=get_settings()
    ds=SupportDataset(args.source,str(s.path("data/raw/tickets.jsonl")))
    df=ds.load(); splits=ds.split(df); save_splits(*splits)
    print(f"Prepared {len(df)} rows into train/val/test splits.")
if __name__=="__main__": main()
