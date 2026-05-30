import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

_chroma_client = chromadb.HttpClient(
    host=settings.chroma_host,
    port=settings.chroma_port,
    settings=ChromaSettings(anonymized_telemetry=False),
)


class VectorStoreService:
    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    def _collection_name(self, tenant_id: Optional[str] = None) -> str:
        # BUG (RAG-08): When tenant_id is None or not provided, this falls back
        # to the "ragcore_shared" collection — a single bucket containing every
        # tenant's documents. Several call sites in rag_chain.py call
        # as_retriever() and query() without passing tenant_id.
        #
        # Any query against ragcore_shared returns chunks from all tenants,
        # leaking confidential documents across organisation boundaries.
        # A user at Company A can receive Company B's internal policy documents
        # in their query results.
        #
        # Fix: remove the default and raise ValueError if tenant_id is None,
        # forcing all call sites to pass an explicit tenant_id.
        if tenant_id:
            return f"ragcore_{tenant_id}"
        return "ragcore_shared"

    def _store(self, tenant_id: Optional[str] = None) -> Chroma:
        return Chroma(
            client=_chroma_client,
            collection_name=self._collection_name(tenant_id),
            embedding_function=self._embeddings,
        )

    def as_retriever(self, tenant_id: Optional[str] = None, k: int = 4):
        return self._store(tenant_id).as_retriever(search_kwargs={"k": k})

    async def query(
        self,
        query_text: str,
        tenant_id: Optional[str] = None,
        k: int = 4,
    ) -> list[Document]:
        store = self._store(tenant_id)

        # BUG (RAG-14): similarity_search_with_relevance_scores() returns
        # (Document, float) pairs where the float is a relevance score in [0, 1].
        # RELEVANCE_THRESHOLD is defined here but the filter is never applied —
        # the list comprehension discards the score and keeps every document.
        # Low-relevance chunks (score 0.1, 0.2) are passed to the LLM alongside
        # high-relevance ones. The LLM then synthesises an answer that mixes
        # relevant content with unrelated passages, producing confident-sounding
        # but factually wrong responses.
        #
        # Fix: docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]
        RELEVANCE_THRESHOLD = 0.72
        results = store.similarity_search_with_relevance_scores(query_text, k=k * 2)
        docs = [doc for doc, score in results]
        return docs[:k]

    async def add_documents(
        self,
        documents: list[Document],
        tenant_id: Optional[str] = None,
    ) -> None:
        store = self._store(tenant_id)
        await store.aadd_documents(documents)
        logger.info("Added %d chunks to collection %s", len(documents), self._collection_name(tenant_id))

    async def delete_document(
        self,
        document_id: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        collection = _chroma_client.get_collection(self._collection_name(tenant_id))
        collection.delete(where={"document_id": document_id})
        logger.info("Deleted chunks for document %s", document_id)

    async def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        # BUG (RAG-07): CrossEncoder.predict() returns raw logits — unbounded
        # floats that can be positive or negative. The sort below uses
        # reverse=False (ascending order), so the document with the LOWEST
        # logit score is returned first — the opposite of the intended behaviour.
        #
        # In a simple test with a single obvious answer, the logit for the correct
        # chunk is highly positive and all others are near zero or negative.
        # Ascending sort puts the correct chunk last. Callers that take [:k] from
        # this result systematically exclude the most relevant document.
        #
        # Fix: reverse=True (descending logits = highest relevance first).
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.page_content) for doc in documents]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=False)
            return [doc for _, doc in ranked]
        except ImportError:
            logger.warning("sentence-transformers not installed; skipping re-rank")
            return documents
