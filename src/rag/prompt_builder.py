from typing import Dict, List

def build_context(chunks: List[Dict]) :
    """Format retrieved chunks as reference data with explicit source metadata."""
    parts=[]
    for i,c in enumerate(chunks,1):
        page=f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[Source {i}: {c.get('document','unknown')}{page}; chunk {c.get('chunk_id')}]\n{c['text']}")
    return "\n\n".join(parts)

def build_sources(chunks: List[Dict]) -> List[Dict]:
    """Expose citation metadata without exposing internal vector details."""
    return [{"document":c.get("document"),"page":c.get("page"),"chunk_id":c.get("chunk_id"),
             "similarity":c.get("similarity")} for c in chunks]
