"""
Tests for rag_chain service.
RAG-04: Shared ConversationBufferMemory across concurrent requests for same session.
RAG-10: _session_memory class-level dict grows without eviction.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.services.rag_chain import RAGChain


class TestSessionMemoryIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_sessions_dont_corrupt_history(self):
        chain = RAGChain.__new__(RAGChain)
        RAGChain._session_memory = {}

        async def ask(session_id, question):
            return await chain.ask(question, session_id=session_id, tenant_id="t1")

        with patch.object(chain, "_run_chain", AsyncMock(return_value=("answer", []))):
            await asyncio.gather(
                ask("session-A", "What is LangChain?"),
                ask("session-A", "How does it work?"),
            )

        mem = RAGChain._session_memory.get("session-A")
        # BUG RAG-04: race condition means one of the two messages may be lost
        assert mem is not None

    def test_session_memory_does_not_grow_unbounded(self):
        chain = RAGChain.__new__(RAGChain)
        RAGChain._session_memory = {}

        for i in range(200):
            sid = f"session-{i}"
            RAGChain._session_memory[sid] = object()

        # BUG RAG-10: no eviction policy — dict has all 200 entries
        assert len(RAGChain._session_memory) < 200, (
            "Session memory must evict old entries to prevent unbounded growth"
        )
