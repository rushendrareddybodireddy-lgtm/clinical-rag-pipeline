"""
/api/v1/alerts — sepsis risk alert endpoints.

Primary endpoint:
    POST /api/v1/alerts/sepsis-risk
        Body: SepsisRiskQuery
        Returns: SepsisAlertResponse

    GET  /api/v1/alerts/sepsis-risk
        Convenience GET with query as a URL param (for quick testing)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.config import get_settings
from api.models.schemas import (
    PatientAlert,
    SepsisAlertResponse,
    SepsisRiskQuery,
    SOFASubScores,
)
from api.rag.llm_client import get_llm_client
from api.rag.prompt_templates import SYSTEM_PROMPT, build_sepsis_alert_messages
from api.rag.retriever import ClinicalNoteRetriever

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

_retriever = ClinicalNoteRetriever()


def _get_pg_conn():
    s = get_settings()
    return psycopg2.connect(
        host=s.postgres_host,
        port=s.postgres_port,
        dbname=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
    )


def _fetch_alert_rows(tier_filter: Optional[list[str]], top_k: int) -> list[dict]:
    """Pull the latest SOFA alert rows from Postgres."""
    conn = _get_pg_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if tier_filter:
            cur.execute(
                """
                SELECT icustay_id, sofa_total, risk_tier,
                       sofa_resp, sofa_coag, sofa_liver,
                       sofa_cardio, sofa_cns, sofa_renal,
                       score_window_end
                FROM sepsis_alerts
                WHERE risk_tier = ANY(%s)
                ORDER BY sofa_total DESC
                LIMIT %s;
                """,
                (tier_filter, top_k),
            )
        else:
            cur.execute(
                """
                SELECT icustay_id, sofa_total, risk_tier,
                       sofa_resp, sofa_coag, sofa_liver,
                       sofa_cardio, sofa_cns, sofa_renal,
                       score_window_end
                FROM sepsis_alerts
                ORDER BY sofa_total DESC
                LIMIT %s;
                """,
                (top_k,),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _record_query_metric(request: Request, query_id: str, body: dict) -> None:
    """Fire-and-forget metric record — wrapped so errors don't fail the request."""
    try:
        conn = _get_pg_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_metrics
                (query_id, query_text, provider, model, latency_ms,
                 tokens_used, estimated_cost_usd, patients_returned, queried_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING;
            """,
            (
                body["query_id"], body["query_text"], body["provider"],
                body["model"], body["latency_ms"], body["tokens_used"],
                body["estimated_cost_usd"], body["patients_returned"],
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning("Failed to record query metric: %s", e)


async def _run_rag_query(payload: SepsisRiskQuery) -> SepsisAlertResponse:
    settings  = get_settings()
    query_id  = str(uuid.uuid4())
    t_start   = time.perf_counter()

    top_k = payload.top_k or settings.top_k_retrieval

    # 1. Pull high-risk patients from Postgres (pre-computed by the SOFA DAG)
    alert_rows = _fetch_alert_rows(payload.risk_tier_filter, top_k)

    if not alert_rows:
        return SepsisAlertResponse(
            query=payload.query,
            provider=payload.llm_provider or settings.llm_provider.value,
            model="n/a",
            patients_at_risk=0,
            alerts=[],
            retrieved_at=datetime.now(timezone.utc),
            latency_ms=0.0,
        )

    # 2. Retrieve relevant note excerpts scoped to high-risk ICU stays
    icustay_ids = [r["icustay_id"] for r in alert_rows]
    try:
        note_excerpts = _retriever.retrieve(
            query=payload.query,
            top_k=min(top_k * 2, 20),
            icustay_ids=icustay_ids,
        )
    except Exception:
        note_excerpts = []

    # 3. Build prompt and call LLM
    messages = build_sepsis_alert_messages(payload.query, alert_rows, note_excerpts)
    llm = get_llm_client(payload.llm_provider)
    llm_response = await llm.chat(messages, system=SYSTEM_PROMPT)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # 4. Parse LLM output into structured alerts
    #    The LLM returns free-text; we attach it as a summary to each patient.
    #    In production you'd parse the structured list more carefully.
    alerts = [
        PatientAlert(
            icustay_id=row["icustay_id"],
            sofa_total=row["sofa_total"],
            risk_tier=row["risk_tier"],
            sub_scores=SOFASubScores(
                respiratory=row["sofa_resp"],
                coagulation=row["sofa_coag"],
                liver=row["sofa_liver"],
                cardiovascular=row["sofa_cardio"],
                cns=row["sofa_cns"],
                renal=row["sofa_renal"],
            ),
            score_window_end=row["score_window_end"],
            summary=llm_response.text,
            supporting_notes=[
                n["chunk_text"][:300]
                for n in note_excerpts
                if n.get("icustay_id") == row["icustay_id"]
            ][:3],
        )
        for row in alert_rows
    ]

    response = SepsisAlertResponse(
        query=payload.query,
        provider=llm_response.provider,
        model=llm_response.model,
        patients_at_risk=len(alerts),
        alerts=alerts,
        retrieved_at=datetime.now(timezone.utc),
        latency_ms=round(latency_ms, 2),
        tokens_used=llm_response.total_tokens,
        estimated_cost_usd=round(llm_response.estimated_cost_usd(), 6),
    )

    # 5. Record metrics asynchronously
    _record_query_metric(None, query_id, {
        "query_id":           query_id,
        "query_text":         payload.query,
        "provider":           llm_response.provider,
        "model":              llm_response.model,
        "latency_ms":         latency_ms,
        "tokens_used":        llm_response.total_tokens,
        "estimated_cost_usd": llm_response.estimated_cost_usd(),
        "patients_returned":  len(alerts),
    })

    return response


@router.post("/sepsis-risk", response_model=SepsisAlertResponse, summary="Sepsis risk alert query")
async def sepsis_risk_post(payload: SepsisRiskQuery, request: Request):
    """
    Submit a natural language clinical query and receive a ranked list of
    ICU patients at elevated sepsis risk with LLM-generated clinical reasoning.
    """
    try:
        return await _run_rag_query(payload)
    except Exception as e:
        log.exception("Error processing sepsis risk query")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sepsis-risk", response_model=SepsisAlertResponse, summary="Sepsis risk alert (GET)")
async def sepsis_risk_get(
    request: Request,
    query: str = Query(
        default="What patients are high risk for sepsis in the next 6 hours?",
        description="Clinical query",
    ),
    top_k: Optional[int] = Query(default=None, ge=1, le=50),
    provider: Optional[str] = Query(default=None, description="claude | openai"),
):
    payload = SepsisRiskQuery(query=query, top_k=top_k, llm_provider=provider)
    try:
        return await _run_rag_query(payload)
    except Exception as e:
        log.exception("Error processing sepsis risk query")
        raise HTTPException(status_code=500, detail=str(e))
