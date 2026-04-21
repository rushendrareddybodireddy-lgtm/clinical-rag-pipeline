"""
pgvector upsert loader.

Handles the write path from the embedding pipeline into Postgres.
The `clinical_notes_embeddings` table uses pgvector's `vector` type for
fast approximate nearest-neighbour search via HNSW indexing.

Schema is defined in infra/postgres/init.sql — this module just writes rows.

Usage:
    loader = PgVectorLoader()
    chunks  = chunk_note(text, metadata)
    vectors = embedder.embed_batch([c["text"] for c in chunks])
    loader.upsert(chunks, vectors)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Sequence

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

log = logging.getLogger(__name__)

_DSN = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    dbname=os.getenv("POSTGRES_DB", "clinical_rag"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
)


@contextmanager
def _get_conn() -> Generator[PgConnection, None, None]:
    conn = psycopg2.connect(**_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class PgVectorLoader:
    """
    Writes (chunk_text, embedding_vector, metadata) rows into `clinical_notes_embeddings`.

    Conflict strategy: ON CONFLICT (hadm_id, chunk_index) DO UPDATE
    This means re-embedding a note will overwrite its vectors cleanly.
    """

    UPSERT_SQL = """
        INSERT INTO clinical_notes_embeddings
            (subject_id, hadm_id, icustay_id, note_category, charttime,
             chunk_index, chunk_text, embedding, metadata)
        VALUES
            (%(subject_id)s, %(hadm_id)s, %(icustay_id)s, %(note_category)s,
             %(charttime)s, %(chunk_index)s, %(chunk_text)s,
             %(embedding)s::vector, %(metadata)s)
        ON CONFLICT (hadm_id, chunk_index)
        DO UPDATE SET
            chunk_text   = EXCLUDED.chunk_text,
            embedding    = EXCLUDED.embedding,
            note_category = EXCLUDED.note_category,
            charttime    = EXCLUDED.charttime,
            updated_at   = NOW();
    """

    def upsert(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> int:
        """
        Upsert chunks with their embedding vectors.

        Args:
            chunks:  list of dicts from chunker.chunk_note()
            vectors: parallel list of embedding vectors

        Returns:
            Number of rows written.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length")

        if not chunks:
            return 0

        rows = []
        for chunk, vec in zip(chunks, vectors):
            # pgvector expects '[0.1,0.2,...]' string format
            vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

            # Anything not a core column goes into metadata JSONB
            meta_keys = {"text", "chunk_index", "char_start", "hadm_id", "icustay_id",
                         "subject_id", "category", "charttime"}
            extra_meta = {k: v for k, v in chunk.items() if k not in meta_keys}

            rows.append({
                "subject_id":    chunk.get("subject_id"),
                "hadm_id":       chunk.get("hadm_id"),
                "icustay_id":    chunk.get("icustay_id"),
                "note_category": chunk.get("category"),
                "charttime":     chunk.get("charttime"),
                "chunk_index":   chunk.get("chunk_index", 0),
                "chunk_text":    chunk["text"],
                "embedding":     vec_str,
                "metadata":      json.dumps(extra_meta),
            })

        with _get_conn() as conn:
            cur = conn.cursor()
            psycopg2.extras.execute_batch(cur, self.UPSERT_SQL, rows, page_size=100)

        log.debug("PgVectorLoader: upserted %d chunks", len(rows))
        return len(rows)

    def backfill_from_silver(self, silver_path: str, batch_size: int = 200) -> None:
        """
        One-shot backfill: read clinical_notes from Silver Delta and embed all of them.
        Useful for initial population of a fresh pgvector database.

        Invoked via: make embed-notes  (which calls python -m embeddings.pgvector_loader --backfill)
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from embeddings.chunker import chunk_note
        from embeddings.embedder import Embedder

        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("pgvector_backfill")
            .master("local[4]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .getOrCreate()
        )

        notes_df = spark.read.format("delta").load(f"{silver_path}/clinical_notes")
        total = notes_df.count()
        log.info("Backfill: %d notes to embed", total)

        embedder = Embedder()
        processed = 0

        # Collect in batches to avoid OOM
        for partition in notes_df.rdd.toLocalIterator():
            row = partition
            chunks = chunk_note(row["TEXT"], note_metadata={
                "hadm_id":    row["HADM_ID"],
                "icustay_id": row.get("ICUSTAY_ID"),
                "subject_id": row["SUBJECT_ID"],
                "category":   row["CATEGORY"],
                "charttime":  str(row.get("CHARTTIME", "")),
            })
            if not chunks:
                continue

            vectors = embedder.embed_batch([c["text"] for c in chunks])
            self.upsert(chunks, vectors)
            processed += 1

            if processed % 100 == 0:
                log.info("Backfill progress: %d / %d notes", processed, total)

        spark.stop()
        log.info("Backfill complete — embedded %d notes.", processed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Run full Silver backfill")
    parser.add_argument(
        "--silver-path",
        default=os.getenv("SILVER_PATH", "/opt/data/silver"),
        help="Path to Silver Delta tables",
    )
    args = parser.parse_args()

    if args.backfill:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        loader = PgVectorLoader()
        loader.backfill_from_silver(args.silver_path)
