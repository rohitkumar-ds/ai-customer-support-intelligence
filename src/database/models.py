from __future__ import annotations
from datetime import datetime, timezone
import uuid
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): pass

def uid(): return str(uuid.uuid4())

class Ticket(Base):
    __tablename__="tickets"
    id: Mapped[str]=mapped_column(String(64),primary_key=True,default=uid)
    created_at: Mapped[datetime]=mapped_column(DateTime,default=lambda:datetime.now(timezone.utc))
    message: Mapped[str]=mapped_column(Text)
    status: Mapped[str]=mapped_column(String(32),default="open")
    response: Mapped[str|None]=mapped_column(Text,nullable=True)
    confidence: Mapped[str|None]=mapped_column(String(16),nullable=True)
    escalated: Mapped[bool]=mapped_column(Boolean,default=False)
    escalation_reason: Mapped[str|None]=mapped_column(Text,nullable=True)
    triage_records=relationship("TriageRecord",back_populates="ticket",cascade="all, delete-orphan")
    rag_logs=relationship("RAGLog",back_populates="ticket",cascade="all, delete-orphan")
    sources=relationship("Source",back_populates="ticket",cascade="all, delete-orphan")

class TriageRecord(Base):
    __tablename__="triage_records"
    id: Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    ticket_id: Mapped[str]=mapped_column(ForeignKey("tickets.id"))
    intent: Mapped[str]=mapped_column(String(100))
    intent_confidence: Mapped[float]=mapped_column(Float)
    sentiment: Mapped[str]=mapped_column(String(32))
    sentiment_confidence: Mapped[float]=mapped_column(Float)
    priority: Mapped[str]=mapped_column(String(32))
    department: Mapped[str]=mapped_column(String(64))
    model_name: Mapped[str]=mapped_column(String(100))
    inference_time_ms: Mapped[float]=mapped_column(Float)
    ticket=relationship("Ticket",back_populates="triage_records")

class RAGLog(Base):
    __tablename__="rag_logs"
    id: Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    ticket_id: Mapped[str]=mapped_column(ForeignKey("tickets.id"))
    query_text: Mapped[str]=mapped_column(Text)
    top_similarity_score: Mapped[float|None]=mapped_column(Float,nullable=True)
    response_draft: Mapped[str|None]=mapped_column(Text,nullable=True)
    retrieval_latency_ms: Mapped[float|None]=mapped_column(Float,nullable=True)
    generation_latency_ms: Mapped[float|None]=mapped_column(Float,nullable=True)
    ticket=relationship("Ticket",back_populates="rag_logs")

class Source(Base):
    __tablename__="sources"
    id: Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    ticket_id: Mapped[str]=mapped_column(ForeignKey("tickets.id"))
    document: Mapped[str]=mapped_column(String(255))
    page: Mapped[int|None]=mapped_column(Integer,nullable=True)
    chunk_id: Mapped[str]=mapped_column(String(255))
    similarity: Mapped[float]=mapped_column(Float)
    ticket=relationship("Ticket",back_populates="sources")

class Conversation(Base):
    __tablename__="conversations"
    id: Mapped[str]=mapped_column(String(64),primary_key=True,default=uid)
    created_at: Mapped[datetime]=mapped_column(DateTime,default=lambda:datetime.now(timezone.utc))

class Feedback(Base):
    __tablename__="feedback"
    id: Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    ticket_id: Mapped[str]=mapped_column(ForeignKey("tickets.id"))
    rating: Mapped[int]=mapped_column(Integer)
    notes: Mapped[str|None]=mapped_column(Text,nullable=True)
