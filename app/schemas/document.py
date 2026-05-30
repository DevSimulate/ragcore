from pydantic import BaseModel
from datetime import datetime


class IngestResponse(BaseModel):
    document_id: str
    status: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunk_count: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
