from __future__ import annotations
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from src.database.models import Ticket,TriageRecord,RAGLog,Source,Feedback

class TicketRepository:
    """Small repository covering ticket persistence and dashboard aggregates."""

    def __init__(self,session:Session): 
        self.session=session

    def create_ticket(self,message:str,status="open",**kwargs)->Ticket:
        """Create and flush a support ticket."""
        ticket=Ticket(message=message,status=status,**kwargs) 
        self.session.add(ticket) 
        self.session.flush() 
        return ticket

    def get_ticket_by_id(self,ticket_id:str) :
        """Fetch one ticket with triage, RAG and source records."""
        return self.session.query(Ticket).options(joinedload(Ticket.triage_records),
            joinedload(Ticket.rag_logs),joinedload(Ticket.sources)).filter(Ticket.id==ticket_id).first()

    def list_tickets(self,limit=50,offset=0,status=None):
        """Return recent tickets with optional status filtering."""
        q=self.session.query(Ticket).order_by(Ticket.created_at.desc())
        if status: q=q.filter(Ticket.status==status)
        return q.offset(offset).limit(limit).all()

    def add_triage_record(self,ticket_id,intent,intent_confidence,sentiment,sentiment_confidence,
                          priority,department,model_name,inference_time_ms):
        """Persist an explainable ticket triage event."""
        row=TriageRecord(ticket_id=ticket_id,intent=intent,intent_confidence=intent_confidence,
            sentiment=sentiment,sentiment_confidence=sentiment_confidence,priority=priority,
            department=department,model_name=model_name,inference_time_ms=inference_time_ms)
        self.session.add(row) 
        self.session.flush() 
        return row

    def add_rag_log(self,ticket_id,query_text,top_similarity_score=None,response_draft=None,
                    retrieval_latency_ms=None,generation_latency_ms=None):
        """Persist retrieval/generation timing and the generated draft."""
        row=RAGLog(ticket_id=ticket_id,query_text=query_text,top_similarity_score=top_similarity_score,
                   response_draft=response_draft,retrieval_latency_ms=retrieval_latency_ms,
                   generation_latency_ms=generation_latency_ms)
        self.session.add(row) 
        self.session.flush()
        return row

    def add_source(self,ticket_id,document,page,chunk_id,similarity):
        """Persist one source citation attached to a ticket."""
        row=Source(ticket_id=ticket_id,document=document,page=page,chunk_id=chunk_id,similarity=similarity)
        self.session.add(row)
        self.session.flush() 
        return row

    def get_analytics_summary(self):
        """Aggregate counts useful to the Streamlit analytics tab."""
        total=self.session.query(func.count(Ticket.id)).scalar() or 0
        triage=self.session.query(TriageRecord).all()
        return {"total_tickets":total,
                "intent_distribution":_counts([x.intent for x in triage]),
                "sentiment_distribution":_counts([x.sentiment for x in triage]),
                "priority_distribution":_counts([x.priority for x in triage]),
                "department_distribution":_counts([x.department for x in triage]),
                "escalated":self.session.query(func.count(Ticket.id)).filter(Ticket.escalated.is_(True)).scalar() or 0,
                "average_confidence":round(sum(x.intent_confidence for x in triage)/len(triage),4) if triage else 0.0}

def _counts(values): 
    out={}
    for v in values: out[v]=out.get(v,0)+1
    return out
