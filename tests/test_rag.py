from src.rag.prompt_builder import build_context

def test_context(): 
    assert "refund.md" in build_context([{"document":"refund.md","page":2,"chunk_id":"x","text":"5-7 days","similarity":.9}])
