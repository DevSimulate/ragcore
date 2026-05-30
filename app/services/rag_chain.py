import logging
from typing import Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGChainService:
    # BUG (RAG-10): _session_memory is a CLASS-level dict, not instance-level.
    # Every RAGChainService object created across all requests shares this single
    # dict. Sessions are never evicted — it grows for every unique session_id
    # that has ever hit the server. On a production instance serving thousands of
    # daily users, this dict holds ConversationBufferMemory objects (each storing
    # full message history) for every session since the last restart.
    # The fix: move to an instance variable AND add an LRU eviction policy,
    # e.g. using functools.lru_cache or a TTL-aware dict limited to N sessions.
    _session_memory: dict[str, ConversationBufferMemory] = {}

    def __init__(self, vector_store: VectorStoreService, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_store = vector_store
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
        )

    def _get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self._session_memory:
            self._session_memory[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        return self._session_memory[session_id]

    def _build_chain(self, session_id: str) -> ConversationalRetrievalChain:
        memory = self._get_or_create_memory(session_id)
        retriever = self.vector_store.as_retriever(tenant_id=self.tenant_id)

        # BUG (RAG-04): A new chain is built per call, but it receives the memory
        # object from the class-level dict (shared across all requests). When two
        # concurrent requests arrive for the same session_id, both retrieve the
        # same ConversationBufferMemory instance. Both chains call memory.save_context()
        # at roughly the same time — the second write overwrites the first, or both
        # append their message pairs in interleaved order. The conversation history
        # becomes corrupted: answers from request A appear attributed to request B's
        # question. The fix: use an asyncio.Lock keyed by session_id to serialise
        # concurrent access to the same session's memory object.
        return ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )

    async def ask(self, question: str, session_id: str) -> dict:
        chain = self._build_chain(session_id)
        result = await chain.ainvoke({"question": question})

        return {
            "answer": result["answer"],
            "session_id": session_id,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in result.get("source_documents", [])
            ],
        }

    def clear_session(self, session_id: str) -> None:
        self._session_memory.pop(session_id, None)
