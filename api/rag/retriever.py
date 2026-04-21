"""
pgvector retriever — similarity search over embedded clinical notes.

Given a query string:
  1. Embed it using the same model used at ingest time.
  2. Run a cosine similarity search against the `clinical_notes_embeddings` table.
  3. Optionally filter by ICU stay IDs (to retrieve notes only for high-risk patients).

We use raw psycopg2 here (rather than SQLAlchemy ORM) because pgvector's
`<=>` operator isn't well-supported in SQLAlchemy's expression language yet.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from api.config import get_settings
from embeddings.embedder import Embedder

log = logging.getLogger(__name__)

_embedder: Optional[Embedder] = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _dsn() -> dict:
    s = get_settings()
    return dict(
        host=s.postgres_host,
        port=s.postgres_port,
        dbname=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
    )


class ClinicalNoteRetriever:
    """
    Retrieves the most relevant clinical note chunks for a given query,
    optionally scoped to a specific set of ICU stay IDs.
    """

    # HNSW cosine distance search — returns rows ordered by distance (lower = more similar)
    SEARCH_SQL = """
        SELECT
            e.icustay_id,
            e.hadm_id,
            e.subject_id,
            e.note_category,
            e.charttime,
            e.chunk_text,
            e.chunk_index,
            1 - (e.embedding <=> %(query_vec)s::vector) AS cosine_similarity
        FROM clinical_notes_embeddings e
        {icustay_filter}
        ORDER BY e.embedding <=> %(query_vec)s::vector
        LIMIT %(top_k)s;
    """

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        icustay_ids: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        """
        Args:
            query:       Natural language query to embed and search
            top_k:       Number of chunks to return (defaults to config value)
            icustay_ids: If provided, restrict search to notes from these stays

        Returns:
            List of dicts with keys: icustay_id, hadm_id, note_category,
            chunk_text, cosine_similarity, etc.
        """
        settings = get_settings()
        k = top_k or settings.top_k_retrieval

        query_embedding = _get_embedder().embed_query(query)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"

        if icustay_ids:
            filter_clause = f"WHERE e.icustay_id = ANY(%(icustay_ids)s)"
        else:
            filter_clause = ""

        sql = self.SEARCH_SQL.format(icustay_filter=filter_clause)

        params: dict[str, Any] = {"query_vec": vec_str, "top_k": k}
        if icustay_ids:
            params["icustay_ids"] = icustay_ids

        conn = psycopg2.connect(**_dsn())
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

        log.debug(
            "Retriever: query='%.60s...' top_k=%d icustay_filter=%s → %d results",
            query, k, bool(icustay_ids), len(rows),
        )
        return rows
