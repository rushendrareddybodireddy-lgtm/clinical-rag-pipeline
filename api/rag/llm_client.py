"""
Switchable LLM client — Claude (Anthropic) or OpenAI.

The active provider is determined by the LLM_PROVIDER env var (or per-request
override).  Both providers implement the same interface so the rest of the
codebase doesn't need to branch on provider type.

Usage:
    client = get_llm_client()               # uses config default
    client = get_llm_client("openai")       # explicit override
    response = await client.chat(messages, system=SYSTEM_PROMPT)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from api.config import LLMProvider, get_settings
from api.rag.prompt_templates import SYSTEM_PROMPT

log = logging.getLogger(__name__)


class LLMResponse:
    __slots__ = ("text", "tokens_input", "tokens_output", "model", "provider")

    def __init__(
        self,
        text: str,
        tokens_input: int,
        tokens_output: int,
        model: str,
        provider: str,
    ) -> None:
        self.text          = text
        self.tokens_input  = tokens_input
        self.tokens_output = tokens_output
        self.model         = model
        self.provider      = provider

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def estimated_cost_usd(self) -> float:
        """Rough cost estimate — update pricing as rates change."""
        pricing = {
            # (input_per_1M, output_per_1M) in USD
            "claude-opus-4-6":        (15.0,  75.0),
            "claude-sonnet-4-6":       (3.0,  15.0),
            "gpt-4o":                  (5.0,  15.0),
            "gpt-4o-mini":             (0.15,  0.6),
        }
        rates = pricing.get(self.model, (5.0, 15.0))
        return (self.tokens_input * rates[0] + self.tokens_output * rates[1]) / 1_000_000


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        ...


class ClaudeClient(BaseLLMClient):
    def __init__(self) -> None:
        import anthropic
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model  = settings.claude_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return LLMResponse(
            text=text,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model=self._model,
            provider="claude",
        )


class OpenAIClient(BaseLLMClient):
    def __init__(self) -> None:
        from openai import AsyncOpenAI
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model  = settings.openai_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        # OpenAI uses a system message in the messages list
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            model=self._model,
            provider="openai",
        )


def get_llm_client(provider_override: Optional[str] = None) -> BaseLLMClient:
    """
    Factory — returns the appropriate client based on config or per-request override.
    """
    settings = get_settings()
    provider = (
        LLMProvider(provider_override)
        if provider_override
        else settings.llm_provider
    )

    if provider == LLMProvider.CLAUDE:
        return ClaudeClient()
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
