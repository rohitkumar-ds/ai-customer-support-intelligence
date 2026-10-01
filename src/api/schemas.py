from __future__ import annotations
from typing import Dict,List,Optional
from pydantic import BaseModel,Field

class MessageRequest(BaseModel):
    message: str = Field(min_length=1,max_length=10000)
    conversation_id: Optional[str]=None

class AnalyzeResponse(BaseModel):
    intent:str; intent_confidence:float
    sentiment:str; sentiment_confidence:float
    entities:Dict
    priority:str; department:str
    latency_ms:float

class Source(BaseModel):
    document:str; page:Optional[int]=None; chunk_id:str; similarity:float

class ChatResponse(AnalyzeResponse):
    answer:str
    sources:List[Source]=[]
    confidence:str
    escalate:bool
    escalation_reason:str=""
    conversation_id:str
    retrieval_latency_ms:float=0
    generation_latency_ms:float=0
    analysis_latency_ms:float=0
    total_latency_ms:float=0
    ticket_id:Optional[str]=None

class RetrieveResponse(BaseModel):
    query:str
    results:List[Source|dict]
