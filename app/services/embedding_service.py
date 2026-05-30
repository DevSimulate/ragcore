import hashlib
import json
import logging

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.cache import redis_client

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 86_400  # 24 hours


class EmbeddingService:
    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    def _cache_key(self, text: str) -> str:
        # BUG (RAG-05): The cache key is derived from the text content only —
        # the model name is not included. When the embedding model is changed
        # (e.g. from text-embedding-3-small to text-embedding-3-large), cached
        # vectors produced by the old model are returned for identical text.
        #
        # text-embedding-3-small produces 1536-dimensional vectors.
        # text-embedding-3-large produces 3072-dimensional vectors.
        #
        # The new ChromaDB collection expects 3072-dim vectors. Inserting 1536-dim
        # cached vectors causes a silent dimension mismatch — ChromaDB either
        # raises an error on insert or silently stores malformed entries, making
        # all similarity searches return garbage results.
        #
        # Fix: include the model name in the key:
        #   f"emb:{settings.embedding_model}:{hashlib.md5(text.encode()).hexdigest()}"
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    async def embed_query(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)

        vector = await self._embeddings.aembed_query(text)
        await redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(vector))
        return vector

    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        # BUG (RAG-13): This calls embed_query() in a loop — one API round-trip
        # per chunk. OpenAIEmbeddings.aembed_documents() accepts a list and sends
        # a single batched request to the API regardless of list length.
        #
        # Ingesting a 200-chunk document makes 200 sequential API calls, each with
        # ~80ms round-trip latency → ~16 seconds total. The batched call takes ~200ms.
        # Under concurrent ingestion this exhausts the rate limit immediately.
        #
        # Fix: check the Redis cache for each text first (keep the per-item cache
        # logic), then send any cache misses as a single aembed_documents() call.
        vectors = []
        for text in texts:
            vector = await self.embed_query(text)
            vectors.append(vector)
        return vectors

    async def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        """Correct batched implementation — used in tests but not wired into the
        ingestion pipeline (see embed_chunks above)."""
        return await self._embeddings.aembed_documents(texts)
