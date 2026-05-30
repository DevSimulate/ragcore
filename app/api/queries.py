import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_tenant
from app.models.query_log import QueryLog
from app.services.rag_chain import RAGChainService
from app.services.vector_store import VectorStoreService
from app.services.llm_service import stream_response
from app.schemas.query import AskRequest, AskResponse, StreamRequest

logger = logging.getLogger(__name__)
router = APIRouter()

_vector_store = VectorStoreService()


def _get_rag_service(tenant_id: str) -> RAGChainService:
    return RAGChainService(vector_store=_vector_store, tenant_id=tenant_id)


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant),
):
    service = _get_rag_service(tenant_id)
    session_id = body.session_id or str(uuid.uuid4())

    result = await service.ask(body.question, session_id=session_id)

    log = QueryLog(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        session_id=session_id,
        question=body.question,
        answer=result["answer"],
        source_count=len(result["source_documents"]),
        created_at=datetime.utcnow(),
    )
    db.add(log)
    await db.commit()

    return AskResponse(
        answer=result["answer"],
        session_id=session_id,
        sources=result["source_documents"],
    )


@router.post("/stream")
async def stream_ask(
    body: StreamRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    docs = await _vector_store.query(
        body.question,
        tenant_id=tenant_id,
        k=4,
    )

    context = "\n\n".join(doc.page_content for doc in docs)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using only "
                "the provided context. If the context does not contain the answer, "
                "say so.\n\nContext:\n" + context
            ),
        },
        {"role": "user", "content": body.question},
    ]

    async def event_stream():
        async for token in stream_response(messages):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
