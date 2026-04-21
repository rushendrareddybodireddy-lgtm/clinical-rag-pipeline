-- Clinical RAG Database Schema
-- Runs once on first Postgres container start via docker-entrypoint-initdb.d/
--
-- Tables:
--   clinical_notes_embeddings  : pgvector store for embedded clinical note chunks
--   sepsis_alerts              : latest SOFA scores per ICU stay (written by Airflow DAG)
--   embedding_queue            : tracks which notes still need embedding
--   query_metrics              : per-request LLMOps tracking

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Airflow DB (separate from application DB) ────────────────────────────────
-- Airflow creates its own DB via the AIRFLOW__CORE__SQL_ALCHEMY_CONN variable.
-- This script only sets up application tables in clinical_rag.

-- ─── Clinical note embeddings ─────────────────────────────────────────────────
-- Stores embedded chunks from NOTEEVENTS (Silver layer).
-- HNSW index for fast approximate nearest-neighbour search.

CREATE TABLE IF NOT EXISTS clinical_notes_embeddings (
    id            BIGSERIAL PRIMARY KEY,
    subject_id    INTEGER,
    hadm_id       INTEGER       NOT NULL,
    icustay_id    INTEGER,
    note_category TEXT,
    charttime     TIMESTAMPTZ,
    chunk_index   INTEGER       NOT NULL DEFAULT 0,
    chunk_text    TEXT          NOT NULL,
    embedding     VECTOR(1536),           -- text-embedding-3-small dimensions
    metadata      JSONB         DEFAULT '{}',
    created_at    TIMESTAMPTZ   DEFAULT NOW(),
    updated_at    TIMESTAMPTZ   DEFAULT NOW(),

    CONSTRAINT uq_hadm_chunk UNIQUE (hadm_id, chunk_index)
);

-- HNSW cosine similarity index — build after data is loaded for speed
CREATE INDEX IF NOT EXISTS idx_notes_embedding_hnsw
    ON clinical_notes_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_notes_icustay
    ON clinical_notes_embeddings (icustay_id);

CREATE INDEX IF NOT EXISTS idx_notes_hadm
    ON clinical_notes_embeddings (hadm_id);

-- ─── Sepsis alerts ────────────────────────────────────────────────────────────
-- One row per ICU stay; upserted hourly by the sofa_scoring DAG.
-- The API reads this table directly at query time — no Spark at request time.

CREATE TABLE IF NOT EXISTS sepsis_alerts (
    icustay_id        INTEGER PRIMARY KEY,
    sofa_total        SMALLINT      NOT NULL,
    risk_tier         TEXT          NOT NULL CHECK (risk_tier IN ('low','moderate','high','critical')),
    sofa_resp         SMALLINT,
    sofa_coag         SMALLINT,
    sofa_liver        SMALLINT,
    sofa_cardio       SMALLINT,
    sofa_cns          SMALLINT,
    sofa_renal        SMALLINT,
    score_window_end  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ   DEFAULT NOW(),
    updated_at        TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_risk_tier
    ON sepsis_alerts (risk_tier, sofa_total DESC);

-- ─── Embedding queue ──────────────────────────────────────────────────────────
-- Populated by the Silver transform job; consumed by the embedding DAG.
-- Using this table avoids a full Delta scan on every embedding run.

CREATE TABLE IF NOT EXISTS embedding_queue (
    queue_id      BIGSERIAL PRIMARY KEY,
    subject_id    INTEGER,
    hadm_id       INTEGER     NOT NULL,
    icustay_id    INTEGER,
    note_category TEXT,
    charttime     TIMESTAMPTZ,
    note_text     TEXT        NOT NULL,
    enqueued_at   TIMESTAMPTZ DEFAULT NOW(),
    embedded_at   TIMESTAMPTZ              -- NULL = pending
);

CREATE INDEX IF NOT EXISTS idx_queue_pending
    ON embedding_queue (enqueued_at)
    WHERE embedded_at IS NULL;

-- ─── Query metrics ────────────────────────────────────────────────────────────
-- LLMOps tracking — one row per API request.

CREATE TABLE IF NOT EXISTS query_metrics (
    query_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text         TEXT,
    provider           TEXT          NOT NULL,
    model              TEXT          NOT NULL,
    latency_ms         DOUBLE PRECISION,
    tokens_used        INTEGER,
    estimated_cost_usd DOUBLE PRECISION,
    patients_returned  INTEGER,
    queried_at         TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_queried_at
    ON query_metrics (queried_at DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_provider
    ON query_metrics (provider, queried_at DESC);

-- ─── Helpful views ────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_high_risk_alerts AS
    SELECT
        sa.*,
        (SELECT COUNT(*) FROM clinical_notes_embeddings cne WHERE cne.icustay_id = sa.icustay_id) AS embedded_note_chunks
    FROM sepsis_alerts sa
    WHERE sa.risk_tier IN ('high', 'critical')
    ORDER BY sa.sofa_total DESC;

CREATE OR REPLACE VIEW v_llm_cost_last_24h AS
    SELECT
        provider,
        COUNT(*)                          AS query_count,
        AVG(latency_ms)                   AS avg_latency_ms,
        SUM(tokens_used)                  AS total_tokens,
        SUM(estimated_cost_usd)           AS total_cost_usd
    FROM query_metrics
    WHERE queried_at >= NOW() - INTERVAL '24 hours'
    GROUP BY provider;
