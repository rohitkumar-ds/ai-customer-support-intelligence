import pytest
from src.knowledge.loader import Document
from src.knowledge.chunker import DocumentChunker

def test_chunking(): 
    assert len(DocumentChunker(10,2).chunk([Document("one two three four five six seven eight nine ten eleven",{"document":"x.md"})]))==2
def test_bad_overlap(): 
    with pytest.raises(ValueError): 
        DocumentChunker(10,10)
