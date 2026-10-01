"""End-to-end retrieval-augmented generation pipeline.

Code flow:
customer question -> query embedding -> FAISS top-K search -> context builder -> LLM
-> qualitative confidence/escalation -> answer + source references.

"""
from __future__ import annotations
import time
from src.config import get_settings
from src.llm.provider import build_llm_client
from src.rag.prompt_builder import build_context, build_sources
from src.rag.verifier import assess_confidence

class RAGPipeline:
    """Coordinate retrieval and grounded response generation."""

    def __init__(self, retriever, llm=None):
        self.retriever=retriever; self.llm=llm or build_llm_client(get_settings())

    def answer(self, query:str, intent_confidence:float=1.0, history=None)->dict:
        """Retrieve evidence, generate a grounded answer, and return citations."""
        settings=get_settings()
        start=time.perf_counter()
        chunks=self.retriever.search(query)
        retrieval_ms=(time.perf_counter()-start)*1000
        if not chunks:
            return {"answer":"I couldn't find sufficient information in the available support knowledge. This case should be reviewed by a support agent.",
                    "sources":[],"confidence":"low","escalate":True,
                    "escalation_reason":"Relevant company knowledge could not be confidently retrieved.",
                    "retrieval_latency_ms":round(retrieval_ms,2)}
        context=build_context(chunks)
        gen_start=time.perf_counter()
        answer=self.llm.generate(query,context,history)
        generation_ms=(time.perf_counter()-gen_start)*1000
        confidence=assess_confidence(intent_confidence,chunks,settings.rag.similarity_threshold)
        return {"answer":answer,"sources":build_sources(chunks),"confidence":confidence,
                "escalate":confidence=="low","escalation_reason":"Evidence or intent confidence is low." if confidence=="low" else "",
                "retrieval_latency_ms":round(retrieval_ms,2),"generation_latency_ms":round(generation_ms,2)}
