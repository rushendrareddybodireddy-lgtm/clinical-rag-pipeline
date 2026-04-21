"""
Application configuration — reads from environment variables.

All config is centralised here so the rest of the codebase only imports
from this module rather than calling os.getenv() scattered everywhere.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"


class Settings(BaseSettings):
    # ─── Database ────────────────────────────────────────────────────────
    postgres_host: str     = "localhost"
    postgres_port: int     = 5432
    postgres_db: str       = "clinical_rag"
    postgres_user: str     = "postgres"
    postgres_password: str = "postgres"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ─── LLM ─────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.CLAUDE
    anthropic_api_key: str    = ""
    openai_api_key: str       = ""
    claude_model: str         = "claude-opus-4-6"
    openai_model: str         = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ─── RAG ─────────────────────────────────────────────────────────────
    top_k_retrieval: int = 10   # patient records returned per query

    # ─── App ─────────────────────────────────────────────────────────────
    log_level: str = "info"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
