from typing import Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class StreamRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceDocument]
