import logging
import os
import tempfile
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db, get_current_tenant
from app.core.config import settings
from app.models.document import Document as DocumentModel
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.schemas.document import DocumentResponse, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_processor = DocumentProcessor()
_vector_store = VectorStoreService()

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _upload_too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"File too large. Maximum upload size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
    )


async def _save_upload_to_temp(file: UploadFile, max_bytes: int) -> tuple[str, int]:
    """Stream an upload to a temp file in chunks. Returns (path, size_bytes)."""
    suffix = Path(file.filename or "").suffix.lower()
    fd, tmp_path = tempfile.mkstemp(prefix="ragcore_", suffix=suffix)
    os.close(fd)

    size = 0
    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise _upload_too_large()
                await out.write(chunk)
    except BaseException:
        _remove_temp_file(tmp_path)
        raise

    return tmp_path, size


def _remove_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)


async def _run_ingestion(
    file_path: str,
    filename: str,
    document_id: str,
    tenant_id: str,
) -> None:
    # BUG (RAG-15, remaining): If document_processor.process_file() raises mid-way (e.g. a
    # corrupt PDF that fails on page 40 of 60), the partially-processed chunks
    # that were already added to ChromaDB are never rolled back. The document
    # record in Postgres shows status="processing" permanently, and the partial
    # chunks are queryable but incomplete.
    #
    # Fix: wrap vector_store.add_documents() and the DB status update in a
    # compensating transaction that deletes partial ChromaDB chunks on failure.
    try:
        chunks = await _processor.process_path(file_path, filename, document_id)
        await _vector_store.add_documents(chunks, tenant_id=tenant_id)

        async with get_db() as db:
            doc = await db.get(DocumentModel, document_id)
            if doc:
                doc.status = "ready"
                doc.chunk_count = len(chunks)
                await db.commit()

        logger.info("Ingestion complete: %s → %d chunks", document_id, len(chunks))
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", document_id, exc)
        try:
            async with get_db() as db:
                doc = await db.get(DocumentModel, document_id)
                if doc:
                    doc.status = "failed"
                    await db.commit()
        except Exception:
            pass
    finally:
        _remove_temp_file(file_path)


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload PDF or DOCX.",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise _upload_too_large()

    document_id = str(uuid.uuid4())
    # file.size may be unknown, so the chunked copy also enforces the limit.
    tmp_path, size_bytes = await _save_upload_to_temp(file, max_bytes)

    try:
        doc = DocumentModel(
            id=document_id,
            filename=file.filename,
            tenant_id=tenant_id,
            size_bytes=size_bytes,
            status="processing",
            chunk_count=0,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
    except BaseException:
        _remove_temp_file(tmp_path)
        raise

    background_tasks.add_task(
        _run_ingestion,
        file_path=tmp_path,
        filename=file.filename,
        document_id=document_id,
        tenant_id=tenant_id,
    )

    return IngestResponse(document_id=document_id, status="processing")


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant),
):
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.tenant_id == tenant_id)
    )
    return result.scalars().all()


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant),
):
    doc = await db.get(DocumentModel, document_id)
    if not doc or doc.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # BUG (RAG-11): The Postgres record is deleted here, but
    # _vector_store.delete_document() is never called. The document's chunks
    # remain in ChromaDB indexed under their document_id and continue to surface
    # in similarity searches. A user who deletes a confidential document to revoke
    # access will still see that document's content returned in query answers.
    #
    # Fix: await _vector_store.delete_document(document_id, tenant_id=tenant_id)
    # before or after the DB delete (ChromaDB delete is idempotent).
    await db.delete(doc)
    await db.commit()
