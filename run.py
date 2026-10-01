"""Convenience launcher for the FastAPI service or Streamlit dashboard."""
import argparse, subprocess, sys

def main():
    """Start the selected application process."""
    p=argparse.ArgumentParser(); p.add_argument("target",choices=["api","dashboard"],default="api",nargs="?")
    args=p.parse_args()
    if args.target=="api":
        subprocess.run([sys.executable,"-m","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"],check=False)
    else:
        subprocess.run([sys.executable,"-m","streamlit","run","dashboard/app.py","--server.port","8501"],check=False)
if __name__=="__main__": main()
