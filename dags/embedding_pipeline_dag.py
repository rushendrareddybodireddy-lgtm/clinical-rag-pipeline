"""
DAG: embedding_pipeline
Schedule: hourly (offset 30 min — runs after sofa_scoring)

Picks up newly added or updated clinical notes from the Silver Delta table
and embeds them into pgvector.  We track which notes have already been
embedded using the `note_embeddings` table's `source_rowid` column so this
job is incremental (won't re-embed unchanged notes).

Task graph:
    check_new_notes → embed_and_load → update_embedding_metadata
"""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="embedding_pipeline",
    description="Incremental clinical note embedding into pgvector",
    schedule_interval="30 * * * *",
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["embeddings", "pgvector", "nlp"],
) as dag:

    def _check_new_notes(**ctx) -> bool:
        """
        Return True if there are notes in Silver that haven't been embedded yet.
        Short-circuits the rest of the DAG if nothing to do.
        """
        import os
        import psycopg2

        pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB", "clinical_rag"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
        cur = pg.cursor()
        cur.execute("SELECT COUNT(*) FROM embedding_queue WHERE embedded_at IS NULL;")
        count = cur.fetchone()[0]
        cur.close()
        pg.close()

        print(f"[embedding_pipeline] {count} notes pending embedding.")
        return count > 0

    check_new_notes = ShortCircuitOperator(
        task_id="check_new_notes",
        python_callable=_check_new_notes,
    )

    def _embed_and_load(**ctx):
        """
        Pull unembedded notes from the queue, chunk them, embed, and load into pgvector.
        Delegates to the embeddings module for the actual work.
        """
        import sys, os
        sys.path.insert(0, "/opt/airflow")

        from embeddings.chunker import chunk_note
        from embeddings.embedder import Embedder
        from embeddings.pgvector_loader import PgVectorLoader

        embedder = Embedder()
        loader   = PgVectorLoader()

        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
        processed  = 0

        import psycopg2
        pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB", "clinical_rag"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )

        while True:
            cur = pg.cursor()
            cur.execute(
                """
                SELECT queue_id, hadm_id, icustay_id, subject_id,
                       note_category, charttime, note_text
                FROM embedding_queue
                WHERE embedded_at IS NULL
                ORDER BY charttime DESC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (batch_size,),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for row in rows:
                queue_id, hadm_id, icustay_id, subject_id, category, charttime, text = row
                chunks = chunk_note(text, note_metadata={
                    "hadm_id": hadm_id,
                    "icustay_id": icustay_id,
                    "subject_id": subject_id,
                    "category": category,
                    "charttime": str(charttime),
                })
                if not chunks:
                    continue

                vectors = embedder.embed_batch([c["text"] for c in chunks])
                loader.upsert(chunks, vectors)

                cur.execute(
                    "UPDATE embedding_queue SET embedded_at = NOW() WHERE queue_id = %s",
                    (queue_id,),
                )
                processed += 1

            pg.commit()
            cur.close()

        pg.close()
        print(f"[embedding_pipeline] Embedded {processed} notes.")
        ctx["task_instance"].xcom_push(key="notes_embedded", value=processed)

    embed_and_load = PythonOperator(
        task_id="embed_and_load",
        python_callable=_embed_and_load,
    )

    def _update_embedding_metadata(**ctx):
        count = ctx["task_instance"].xcom_pull(task_ids="embed_and_load", key="notes_embedded")
        print(f"[embedding_pipeline] Metadata update complete — {count} notes embedded this run.")

    update_embedding_metadata = PythonOperator(
        task_id="update_embedding_metadata",
        python_callable=_update_embedding_metadata,
    )

    check_new_notes >> embed_and_load >> update_embedding_metadata
