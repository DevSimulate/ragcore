"""
Tests for vector_store service.
RAG-07: Results sorted ascending (worst first).
RAG-08: None tenant_id falls back to shared collection.
RAG-14: Relevance threshold defined but never applied.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.vector_store import VectorStoreService


@pytest.fixture
def store():
    return VectorStoreService()


class TestSortOrder:
    @pytest.mark.asyncio
    async def test_top_result_has_highest_score(self, store):
        docs = [
            (MagicMock(page_content="irrelevant"), 0.31),
            (MagicMock(page_content="most relevant"), 0.95),
            (MagicMock(page_content="somewhat relevant"), 0.62),
        ]
        store._collection = MagicMock()
        with patch.object(store, "_similarity_search_with_score", AsyncMock(return_value=docs)):
            results = await store.search("query", "tenant1", top_k=3)

        assert results[0].page_content == "most relevant", (
            "RAG-07: ascending sort puts worst document first"
        )


class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_none_tenant_raises_not_falls_back(self, store):
        with pytest.raises(ValueError, match="tenant_id"):
            await store.search("query", None, top_k=5)
        # BUG RAG-08: falls back to "ragcore_shared" instead of raising


class TestRelevanceThreshold:
    @pytest.mark.asyncio
    async def test_low_score_docs_filtered_out(self, store):
        docs = [
            (MagicMock(page_content="good"), 0.91),
            (MagicMock(page_content="borderline"), 0.71),
            (MagicMock(page_content="noise"), 0.40),
        ]
        with patch.object(store, "_similarity_search_with_score", AsyncMock(return_value=docs)):
            results = await store.search("query", "tenant1", top_k=3)

        contents = [r.page_content for r in results]
        # BUG RAG-14: threshold constant exists but is never used in filter
        assert "noise" not in contents, "Documents below relevance threshold must be excluded"
