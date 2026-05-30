import asyncio
import logging
from typing import AsyncGenerator

import openai
import tiktoken
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)
_encoder = tiktoken.encoding_for_model("gpt-4o")

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-3.5-turbo": 16_385,
}

MAX_RETRIES = 3


def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_encoder.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """Return total token count across a list of chat messages."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # per-message formatting overhead
    return total + 2  # reply priming tokens


async def chat_complete(messages: list[dict], temperature: float = 0.1) -> str:
    # BUG (RAG-01): count_messages_tokens() is defined above but is never called
    # before this API request. When a large document is assembled into the context
    # window and exceeds the model limit, OpenAI returns a 400 InvalidRequestError:
    # "This model's maximum context length is 128000 tokens. Your messages resulted
    # in X tokens." The fix is to call count_messages_tokens(messages) here and
    # truncate or raise a user-friendly error before hitting the API.
    response = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


async def stream_response(messages: list[dict]) -> AsyncGenerator[str, None]:
    # BUG (RAG-02): No context window check before the stream opens.
    # OpenAI begins streaming tokens immediately, then aborts mid-response
    # if the total context exceeds the model limit. The HTTP connection drops
    # with no error frame — the client receives a partial answer followed by
    # a broken pipe. The fix: call count_messages_tokens(messages) and verify
    # against MODEL_CONTEXT_LIMITS[settings.llm_model] before starting the stream.
    async for chunk in await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        stream=True,
        temperature=0.1,
    ):
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


async def chat_complete_with_retry(messages: list[dict]) -> str:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return await chat_complete(messages)
        except openai.RateLimitError as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                raise

            # BUG (RAG-09): OpenAI 429 responses carry a Retry-After header
            # that specifies the exact number of seconds to wait. This code
            # ignores that header entirely and uses a fixed exponential backoff
            # (1 s, 2 s, 4 s). Under a sustained burst this means retrying too
            # early and burning retry budget. The fix:
            #   retry_after = int(e.response.headers.get("retry-after", 2 ** attempt))
            #   await asyncio.sleep(retry_after)
            wait_seconds = 2 ** attempt
            logger.warning(
                "Rate limited by OpenAI (attempt %d/%d). Sleeping %ds.",
                attempt + 1,
                MAX_RETRIES,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)

        except openai.APIStatusError as e:
            raise

    raise last_error  # type: ignore[misc]
