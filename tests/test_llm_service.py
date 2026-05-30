"""
Tests for llm_service.
RAG-01: Token count not checked before API call.
RAG-02: Streaming skips context window guard.
RAG-09: Retry logic ignores Retry-After header.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.llm_service import LLMService


@pytest.fixture
def svc():
    return LLMService()


class TestTokenGuard:
    @pytest.mark.asyncio
    async def test_raises_when_messages_exceed_context(self, svc):
        messages = [{"role": "user", "content": "x " * 200_000}]
        with pytest.raises(ValueError, match="context"):
            await svc.chat_complete(messages)
        # BUG RAG-01: no token check → sends request, gets 400 from OpenAI

    @pytest.mark.asyncio
    async def test_stream_raises_when_messages_exceed_context(self, svc):
        messages = [{"role": "user", "content": "y " * 200_000}]
        with pytest.raises(ValueError, match="context"):
            async for _ in svc.stream_response(messages):
                pass
        # BUG RAG-02: stream_response skips the guard entirely


class TestRetryAfterHeader:
    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self, svc):
        import time
        from openai import RateLimitError

        retry_after = 1  # seconds
        error = RateLimitError.__new__(RateLimitError)
        error.response = MagicMock()
        error.response.headers = {"Retry-After": str(retry_after)}

        call_times = []

        async def flaky(*args, **kwargs):
            call_times.append(time.monotonic())
            if len(call_times) < 2:
                raise error
            return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

        with patch.object(svc._client.chat.completions, "create", AsyncMock(side_effect=flaky)):
            await svc.chat_complete_with_retry([{"role": "user", "content": "hi"}])

        elapsed = call_times[1] - call_times[0]
        # BUG RAG-09: uses 2**attempt backoff, Retry-After header ignored
        assert elapsed >= retry_after * 0.9, "Must honor Retry-After header"
