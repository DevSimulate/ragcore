"""
Tests for document_processor service.
RAG-03: CharacterTextSplitter produces uneven chunks on real prose.
RAG-06: Image-only PDF pages are silently skipped.
RAG-12: Concurrent uploads with same document_id produce duplicate chunks.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


class TestChunkSplitting:
    def test_splits_long_prose_into_even_chunks(self, processor):
        text = " ".join(["word"] * 500)
        chunks = processor.split_text(text)
        # With RecursiveCharacterTextSplitter all chunks should be ≤ chunk_size
        # BUG RAG-03: CharacterTextSplitter on "\n" produces 1 giant chunk here
        assert all(len(c) <= 1200 for c in chunks), (
            "CharacterTextSplitter fails to split prose without newlines"
        )

    def test_chunk_overlap_preserves_context(self, processor):
        text = "alpha " * 300 + "SENTINEL " + "beta " * 300
        chunks = processor.split_text(text)
        sentinel_chunks = [c for c in chunks if "SENTINEL" in c]
        assert len(sentinel_chunks) >= 1


class TestPDFProcessing:
    @pytest.mark.asyncio
    async def test_warns_on_image_only_page(self, processor, caplog):
        import logging
        pdf_bytes = b"%PDF-1.4 fake-image-only-pdf"
        with patch.object(processor, "_extract_text_from_pdf", return_value=[]):
            with caplog.at_level(logging.WARNING):
                result = await processor.process_pdf(pdf_bytes, "scan.pdf")
        # BUG RAG-06: warning is emitted at DEBUG, not WARNING — test catches it
        assert any("image" in r.message.lower() or "no text" in r.message.lower()
                   for r in caplog.records), "Should warn user about image-only pages"


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_concurrent_upload_same_id_no_duplicates(self, processor):
        import asyncio
        doc_id = "doc-abc-123"
        call_count = 0

        async def fake_store(chunks, doc_id, tenant_id):
            nonlocal call_count
            call_count += 1

        processor._store_chunks = fake_store
        processor._already_ingested = AsyncMock(return_value=False)

        await asyncio.gather(
            processor.ingest("text content here", doc_id, "tenant1"),
            processor.ingest("text content here", doc_id, "tenant1"),
        )
        # BUG RAG-12: no idempotency guard → call_count == 2
        assert call_count == 1, "Duplicate ingest for same document_id should be skipped"
