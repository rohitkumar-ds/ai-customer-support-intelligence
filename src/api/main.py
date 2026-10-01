from __future__ import annotations
from pathlib import Path
import time
from fastapi import FastAPI,HTTPException,Query
from src.config import get_settings
from src.api.schemas import MessageRequest,AnalyzeResponse,ChatResponse
from src.models.classical import ClassicalIntentClassifier
from src.sentiment.analyzer import SentimentAnalyzer
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.database.connection import init_db,get_session
from src.database.repository import TicketRepository
from src.services.support_service import SupportService
from src.memory.store import ConversationMemory
memory=ConversationMemory(max_messages=6)

settings=get_settings()
app=FastAPI(title=settings.app.name,version=settings.app.version)

_service=None

def build_service():
    """Construct the application service from configured artifacts."""
    model_path=settings.path(settings.models.classical.artifact_path)
    if not model_path.exists():
        raise RuntimeError("Intent model is missing. Run `python -m src.models.train_classical` first.")
    classifier=ClassicalIntentClassifier.load(str(model_path))
    sentiment=SentimentAnalyzer(settings.sentiment.model_name)
    index=Path(settings.path(settings.rag.index_path))
    if index.exists():
        rag=RAGPipeline(Retriever())
    else:
        class NoKnowledge:
            def search(self,*args,**kwargs): return []
        rag=RAGPipeline(NoKnowledge())
    return classifier,sentiment,rag

@app.on_event("startup")
def startup():
    """Provision database tables at application startup."""
    init_db()

@app.get("/health")
def health():
    """Report whether required core configuration/artifacts are present."""
    model=Path(settings.path(settings.models.classical.artifact_path)).exists()
    index=Path(settings.path(settings.rag.index_path)).exists()
    return {"status":"ok","classifier_artifact":model,"knowledge_index":index,"llm_provider":settings.llm.provider}

@app.post("/analyze",response_model=AnalyzeResponse)
def analyze(request:MessageRequest):
    """Analyze intent, sentiment, entities, priority and department."""
    try: classifier,sentiment,rag=build_service()
    except Exception as exc: raise HTTPException(503,str(exc))
    return SupportService(classifier,sentiment,rag).analyze(request.message)

@app.post("/predict-intent")
def predict_intent(request:MessageRequest):
    """Return only intent classification, useful for model comparison clients."""
    try:
        path=settings.path(settings.models.classical.artifact_path)
        model=ClassicalIntentClassifier.load(str(path))
        return model.predict_one(request.message)
    except Exception as exc: raise HTTPException(503,str(exc))

@app.post("/retrieve")
def retrieve(request:MessageRequest):
    """Search the knowledge base and expose source metadata and similarity."""
    try:
        retriever=Retriever(); return {"query":request.message,"results":retriever.search(request.message)}
    except Exception as exc: raise HTTPException(503,str(exc))

@app.post("/chat",response_model=ChatResponse)
def chat(request:MessageRequest):
    """Run the full support intelligence + grounded RAG workflow."""
    global _service

    request_start=time.perf_counter()

    try:
        if _service is None:
            classifier,sentiment,rag=build_service()
            _service=SupportService(classifier,sentiment,rag)
    except Exception as exc:
        raise HTTPException(503,str(exc))

    with_db=get_session()

    try:
        conversation_id=request.conversation_id or "default"
        history=memory.get(conversation_id)

        _service.repository=TicketRepository(with_db)

        result=_service.chat(
            request.message,
            conversation_id,
            history
        )

        memory.add(conversation_id,"user",request.message)
        memory.add(conversation_id,"assistant",result["answer"])

        with_db.commit()

        # analyze() currently stores its timing in latency_ms.
        # Rename it so the API clearly distinguishes analysis from
        # complete end-to-end request latency.
        result["analysis_latency_ms"]=result["latency_ms"]

        # Measure the complete /chat request, including analysis,
        # retrieval, generation, memory and database persistence.
        result["total_latency_ms"]=round(
            (time.perf_counter()-request_start)*1000,
            2
        )

        return result

    except ValueError as exc:
        with_db.rollback()
        raise HTTPException(400,str(exc))

    except Exception:
        with_db.rollback()
        raise HTTPException(
            500,
            "Support request could not be completed."
        )

    finally:
        with_db.close()

@app.post("/knowledge/ingest")
def knowledge_ingest():
    """Build/rebuild the persistent knowledge index."""
    try:
        from scripts.build_index import main
        main(); return {"status":"ok","message":"Knowledge index rebuilt."}
    except Exception as exc: raise HTTPException(500,str(exc))

@app.get("/tickets")
def tickets(limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
    """List recent persisted tickets."""
    session=get_session()
    try:
        rows=TicketRepository(session).list_tickets(limit,offset)
        return [{"id":x.id,"created_at":x.created_at,"message":x.message,"status":x.status,
                 "confidence":x.confidence,"escalated":x.escalated} for x in rows]
    finally: session.close()

@app.get("/tickets/{ticket_id}")
def ticket(ticket_id:str):
    """Return one ticket and its audit records."""
    session=get_session()
    try:
        row=TicketRepository(session).get_ticket_by_id(ticket_id)
        if not row: raise HTTPException(404,"Ticket not found.")
        return {"id":row.id,"created_at":row.created_at,"message":row.message,"response":row.response,
                "confidence":row.confidence,"escalated":row.escalated,"escalation_reason":row.escalation_reason,
                "triage":[{"intent":t.intent,"confidence":t.intent_confidence,"sentiment":t.sentiment,
                           "priority":t.priority,"department":t.department} for t in row.triage_records],
                "sources":[{"document":s.document,"page":s.page,"chunk_id":s.chunk_id,"similarity":s.similarity} for s in row.sources]}
    finally: session.close()
