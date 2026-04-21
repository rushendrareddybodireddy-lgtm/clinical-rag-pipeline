"""
/metrics — Prometheus exposition endpoint + internal LLMOps metrics router.

Prometheus scrapes /metrics on a regular interval (see prometheus.yml).
We also expose /api/v1/metrics/summary for a human-readable JSON summary.
"""

from __future__ import annotations

import logging

import psycopg2
import psycopg2.extras
from fastapi import APIRouter
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

from api.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])

# ─── Prometheus metrics ────────────────────────────────────────────────────────

rag_query_counter = Counter(
    "clinical_rag_queries_total",
    "Total number of RAG queries processed",
    labelnames=["provider", "risk_tier"],
)

rag_latency_histogram = Histogram(
    "clinical_rag_query_latency_seconds",
    "End-to-end RAG query latency",
    labelnames=["provider"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

llm_tokens_counter = Counter(
    "clinical_rag_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["provider", "token_type"],
)

llm_cost_counter = Counter(
    "clinical_rag_cost_usd_total",
    "Estimated LLM cost in USD",
    labelnames=["provider"],
)

alerts_gauge = Gauge(
    "clinical_rag_active_sepsis_alerts",
    "Current number of patients flagged in the sepsis_alerts table",
    labelnames=["risk_tier"],
)

embedding_queue_gauge = Gauge(
    "clinical_rag_embedding_queue_size",
    "Number of notes pending embedding",
)


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    # Refresh gauges before exposing
    _refresh_gauges()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/v1/metrics/summary", summary="LLMOps dashboard summary")
async def metrics_summary():
    """
    Human-readable JSON summary of LLMOps metrics for the past 24h.
    Grafana is configured to scrape /metrics directly; this endpoint
    is for quick eyeballing from the API docs.
    """
    s = get_settings()
    conn = psycopg2.connect(
        host=s.postgres_host, port=s.postgres_port,
        dbname=s.postgres_db, user=s.postgres_user,
        password=s.postgres_password,
    )

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                provider,
                COUNT(*)                               AS total_queries,
                AVG(latency_ms)                        AS avg_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP
                    (ORDER BY latency_ms)              AS p95_latency_ms,
                SUM(tokens_used)                       AS total_tokens,
                SUM(estimated_cost_usd)                AS total_cost_usd
            FROM query_metrics
            WHERE queried_at >= NOW() - INTERVAL '24 hours'
            GROUP BY provider;
        """)
        usage = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT risk_tier, COUNT(*) AS patient_count
            FROM sepsis_alerts
            GROUP BY risk_tier
            ORDER BY patient_count DESC;
        """)
        alert_breakdown = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT COUNT(*) AS pending FROM embedding_queue WHERE embedded_at IS NULL;
        """)
        queue_size = cur.fetchone()["pending"]

    finally:
        conn.close()

    return {
        "llm_usage_last_24h": usage,
        "sepsis_alerts_by_tier": alert_breakdown,
        "embedding_queue_pending": queue_size,
    }


def _refresh_gauges() -> None:
    """Update Prometheus Gauge metrics from Postgres state."""
    try:
        s = get_settings()
        conn = psycopg2.connect(
            host=s.postgres_host, port=s.postgres_port,
            dbname=s.postgres_db, user=s.postgres_user,
            password=s.postgres_password,
        )
        cur = conn.cursor()

        cur.execute("""
            SELECT risk_tier, COUNT(*) FROM sepsis_alerts GROUP BY risk_tier;
        """)
        for tier, count in cur.fetchall():
            alerts_gauge.labels(risk_tier=tier).set(count)

        cur.execute("SELECT COUNT(*) FROM embedding_queue WHERE embedded_at IS NULL;")
        embedding_queue_gauge.set(cur.fetchone()[0])

        conn.close()
    except Exception as e:
        log.warning("Failed to refresh gauges: %s", e)
