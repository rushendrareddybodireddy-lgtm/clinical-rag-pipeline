"""
Embedding provider abstraction.

Supports OpenAI text-embedding-3-small (default) and a local
sentence-transformers fallback for offline/dev use.

The Airflow embedding DAG and the FastAPI retriever both import this class,
so we keep it dependency-light and stateless (no global model state outside
the class).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

log = logging.getLogger(__name__)

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION    = 1536   # text-embedding-3-small output dim


class Embedder:
    """
    Thin wrapper around OpenAI's embedding API with:
      - automatic batching (API max: 2048 inputs per request)
      - exponential back-off on rate limit errors
      - optional fallback to sentence-transformers if OPENAI_API_KEY is unset
    """

    MAX_BATCH_SIZE = 512   # conservative batch to avoid token-limit errors on long texts
    MAX_RETRIES    = 5
    BASE_DELAY     = 1.0   # seconds

    def __init__(self) -> None:
        self._use_openai = bool(os.getenv("OPENAI_API_KEY"))
        if self._use_openai:
            from openai import OpenAI
            self._client = OpenAI()
            log.info("Embedder: using OpenAI %s", OPENAI_EMBEDDING_MODEL)
        else:
            log.warning(
                "OPENAI_API_KEY not set — falling back to sentence-transformers "
                "(all-MiniLM-L6-v2).  Vectors will NOT match production dimensions."
            )
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of strings, returning one vector per input."""
        if not texts:
            return []

        if self._use_openai:
            return self._embed_openai(list(texts))
        else:
            return self._embed_local(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (used at retrieval time)."""
        return self.embed_batch([text])[0]

    # ─── Private ──────────────────────────────────────────────────────────────

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self._client.embeddings.create(
                        model=OPENAI_EMBEDDING_MODEL,
                        input=batch,
                    )
                    vectors.extend([item.embedding for item in response.data])
                    break
                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    delay = self.BASE_DELAY * (2 ** attempt)
                    log.warning("Embedding API error (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
        return vectors

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._local_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
