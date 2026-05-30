"""
Tests for embedding_service.
RAG-05: Cache key does not include model name — model swap returns stale embeddings.
RAG-13: embed_chunks calls embed_query in a loop instead of batch API.
"""
import pytest
from unittest.mock import AsyncMock, patch, call
from app.services.embedding_service import EmbeddingService


@pytest.fixture
def svc():
    return EmbeddingService()


class TestCacheKey:
    @pytest.mark.asyncio
    async def test_different_models_dont_share_cache(self, svc):
        text = "hello world"
        fake_vec_a = [0.1] * 1536
        fake_vec_b = [0.9] * 3072

        svc._model_name = "text-embedding-3-small"
        with patch.object(svc, "_call_api", AsyncMock(return_value=fake_vec_a)):
            result_a = await svc.embed_text(text)

        svc._model_name = "text-embedding-3-large"
        with patch.object(svc, "_call_api", AsyncMock(return_value=fake_vec_b)):
            result_b = await svc.embed_text(text)

        # BUG RAG-05: cache key is md5(text) only — result_b returns fake_vec_a
        assert result_a != result_b, "Cache must be keyed by (model, text)"


class TestBatchEmbedding:
    @pytest.mark.asyncio
    async def test_embed_chunks_uses_batch_api(self, svc):
        chunks = [f"chunk number {i}" for i in range(10)]
        batch_mock = AsyncMock(return_value=[[float(i)] * 1536 for i in range(10)])

        with patch.object(svc._embeddings, "aembed_documents", batch_mock) as m:
            await svc.embed_chunks(chunks)

        # BUG RAG-13: loops over embed_query → aembed_documents called 0 times
        m.assert_called_once(), "embed_chunks must use aembed_documents batch call"
