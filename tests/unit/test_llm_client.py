"""
Unit tests for the LLM client abstraction layer.

All actual API calls are mocked — these tests verify routing logic,
cost estimation, and provider switching behaviour.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.rag.llm_client import LLMResponse, get_llm_client


class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse("text", 100, 50, "claude-opus-4-6", "claude")
        assert r.total_tokens == 150

    def test_cost_estimation_claude_opus(self):
        r = LLMResponse("text", 1_000_000, 1_000_000, "claude-opus-4-6", "claude")
        # 1M input @ $15 + 1M output @ $75 = $90
        assert abs(r.estimated_cost_usd() - 90.0) < 0.01

    def test_cost_estimation_gpt4o(self):
        r = LLMResponse("text", 1_000_000, 1_000_000, "gpt-4o", "openai")
        # 1M input @ $5 + 1M output @ $15 = $20
        assert abs(r.estimated_cost_usd() - 20.0) < 0.01

    def test_unknown_model_uses_default(self):
        r = LLMResponse("text", 100, 100, "some-unknown-model", "openai")
        cost = r.estimated_cost_usd()
        assert cost > 0


class TestGetLLMClient:
    def test_returns_claude_by_default(self):
        with patch("api.rag.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = MagicMock(value="claude")
            mock_settings.return_value.anthropic_api_key = "sk-ant-test"
            mock_settings.return_value.claude_model = "claude-opus-4-6"

            with patch("anthropic.AsyncAnthropic"):
                client = get_llm_client()
                from api.rag.llm_client import ClaudeClient
                assert isinstance(client, ClaudeClient)

    def test_provider_override_openai(self):
        with patch("api.rag.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-ant-test"
            mock_settings.return_value.openai_api_key = "sk-openai-test"
            mock_settings.return_value.openai_model = "gpt-4o"

            with patch("openai.AsyncOpenAI"):
                client = get_llm_client(provider_override="openai")
                from api.rag.llm_client import OpenAIClient
                assert isinstance(client, OpenAIClient)

    def test_missing_api_key_raises(self):
        with patch("api.rag.llm_client.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = ""
            mock_settings.return_value.llm_provider = MagicMock(value="claude")

            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                get_llm_client()


class TestChunker:
    def test_short_note_returns_single_chunk(self):
        from embeddings.chunker import chunk_note
        text = "Chief Complaint: chest pain. Assessment: rule out MI."
        chunks = chunk_note(text, {"hadm_id": 1, "chunk_index": 0})
        assert len(chunks) >= 1
        assert chunks[0]["text"] == text.strip() or text.strip() in chunks[0]["text"]

    def test_empty_note_returns_empty(self):
        from embeddings.chunker import chunk_note
        assert chunk_note("") == []
        assert chunk_note("   ") == []

    def test_long_note_splits_into_multiple_chunks(self):
        from embeddings.chunker import chunk_note, CHUNK_SIZE_CHARS
        long_text = "This is a clinical note. " * (CHUNK_SIZE_CHARS // 10)
        chunks = chunk_note(long_text, {"hadm_id": 1})
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        from embeddings.chunker import chunk_note, CHUNK_SIZE_CHARS
        text = "word " * (CHUNK_SIZE_CHARS // 3)
        chunks = chunk_note(text, {"hadm_id": 1})
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagated_to_chunks(self):
        from embeddings.chunker import chunk_note
        meta = {"hadm_id": 99, "icustay_id": 42, "category": "Discharge summary"}
        chunks = chunk_note("Some clinical text here.", meta)
        for c in chunks:
            assert c["hadm_id"] == 99
            assert c["icustay_id"] == 42
