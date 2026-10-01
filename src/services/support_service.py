"""Application service that orchestrates customer-support intelligence.

Code flow:
message -> intent -> sentiment -> entities -> priority/routing -> retrieval/RAG
-> escalation -> SQLite audit record.

Individual AI components remain replaceable and testable; this service only coordinates them.
"""
from __future__ import annotations
import time, json, uuid
from typing import Optional

class SupportService:
    """Coordinate analysis, RAG, and persistence for a customer message."""

    def __init__(self,classifier,sentiment_analyzer,rag_pipeline,repository=None):
        self.classifier=classifier; self.sentiment=sentiment_analyzer
        self.rag=rag_pipeline; self.repository=repository

    def analyze(self,message:str)->dict:
        """Run deterministic customer-query analysis without generating an answer."""
        
        from src.entities.extractor import extract_entities
        from src.intelligence.priority import determine_priority
        from src.intelligence.routing import route_department

        if not message or not message.strip():
            raise ValueError("message must not be empty")
        start=time.perf_counter()
        pred=self.classifier.predict_one(message)
        sent=self.sentiment.analyze(message)
        entities=extract_entities(message)
        priority=determine_priority(pred["intent"],sent["sentiment"],pred["confidence"],entities,message)
        result={"intent":pred["intent"],"intent_confidence":round(pred["confidence"],4),
                "sentiment":sent["sentiment"],"sentiment_confidence":round(sent["confidence"],4),
                "entities":entities,"priority":priority,"department":route_department(pred["intent"]),
                "latency_ms":round((time.perf_counter()-start)*1000,2)}
        return result

    def chat(self,message:str,conversation_id:Optional[str]=None,history=None)->dict:
        """Analyze, retrieve knowledge, generate a grounded response and optionally persist it."""
        
        from src.rag.verifier import should_escalate
        
        analysis=self.analyze(message)
        rag=self.rag.answer(message,analysis["intent_confidence"],history)
        from src.config import get_settings
        settings=get_settings()
        policy_escalate,reason=should_escalate(analysis["intent_confidence"],analysis["sentiment"],analysis["intent"],
                                               [{"similarity":s["similarity"]} for s in rag["sources"]],settings)
        escalated=rag["escalate"] or policy_escalate
        result={**analysis,**{k:v for k,v in rag.items() if k not in {"escalate","escalation_reason"}},
                "conversation_id":conversation_id or str(uuid.uuid4()),"escalate":escalated,
                "escalation_reason":reason or rag.get("escalation_reason","")}
        if self.repository:
            ticket=self.repository.create_ticket(message=message,status="open")
            self.repository.add_triage_record(ticket.id,analysis["intent"],analysis["intent_confidence"],
                                              analysis["sentiment"],analysis["sentiment_confidence"],
                                              analysis["priority"],analysis["department"],"configured",analysis["latency_ms"])
            ticket.response=rag["answer"] 
            ticket.confidence=rag["confidence"] 
            ticket.escalated=escalated
            ticket.escalation_reason=result["escalation_reason"]
            for src in rag["sources"]:
                self.repository.add_source(ticket.id,src["document"],src.get("page"),src["chunk_id"],src["similarity"])
            self.repository.add_rag_log(
                ticket.id,
                message,
                top_similarity_score=rag["sources"][0]["similarity"] if rag["sources"] else None,
                response_draft=rag["answer"],
                retrieval_latency_ms=rag.get("retrieval_latency_ms"),
                generation_latency_ms=rag.get("generation_latency_ms"),
            )
            result["ticket_id"]=ticket.id
        return result
