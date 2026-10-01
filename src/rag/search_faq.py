import argparse
from src.config import get_settings
from src.rag.faq_matcher import FAQMatcher

def main():
    """Run a one-query or small demo semantic search."""
    parser=argparse.ArgumentParser() 
    parser.add_argument("--query") 
    parser.add_argument("--demo",action="store_true")
    args=parser.parse_args()
    matcher=FAQMatcher(str(get_settings().path("data/knowledge_base/faqs.jsonl")))
    queries=[args.query] if args.query else (["I was charged twice on my card","How long does a refund take?"] if args.demo else [])
    for q in queries:
        print(f"\nQuery: {q}")
        for r in matcher.match(q):
            print(r["rank"],r["similarity"],r["question"])

if __name__=="__main__": 
    main()
