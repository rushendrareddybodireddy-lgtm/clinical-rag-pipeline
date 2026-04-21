"""
Pydantic request/response schemas for the Clinical RAG API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Alert schemas ─────────────────────────────────────────────────────────────

class SOFASubScores(BaseModel):
    respiratory:    int = Field(..., ge=0, le=4, description="PaO2/FiO2 sub-score")
    coagulation:    int = Field(..., ge=0, le=4, description="Platelet sub-score")
    liver:          int = Field(..., ge=0, le=4, description="Bilirubin sub-score")
    cardiovascular: int = Field(..., ge=0, le=4, description="MAP sub-score")
    cns:            int = Field(..., ge=0, le=4, description="GCS sub-score")
    renal:          int = Field(..., ge=0, le=4, description="Creatinine sub-score")


class PatientAlert(BaseModel):
    icustay_id:       int
    sofa_total:       int    = Field(..., ge=0, le=24)
    risk_tier:        str    = Field(..., description="low | moderate | high | critical")
    sub_scores:       SOFASubScores
    score_window_end: datetime
    summary:          str    = Field(..., description="LLM-generated clinical reasoning summary")
    supporting_notes: list[str] = Field(default_factory=list, description="Retrieved note excerpts")


class SepsisAlertResponse(BaseModel):
    query:              str
    provider:           str    = Field(..., description="LLM provider used: claude | openai")
    model:              str
    patients_at_risk:   int
    alerts:             list[PatientAlert]
    retrieved_at:       datetime
    latency_ms:         float
    tokens_used:        Optional[int] = None
    estimated_cost_usd: Optional[float] = None


# ─── Query schemas ─────────────────────────────────────────────────────────────

class SepsisRiskQuery(BaseModel):
    query: str = Field(
        default="What patients are high risk for sepsis in the next 6 hours?",
        description="Natural language query from the clinician",
        max_length=1000,
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default retrieval count (default set in config)",
    )
    risk_tier_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter results to specific tiers: low, moderate, high, critical",
    )
    llm_provider: Optional[str] = Field(
        default=None,
        description="Override LLM provider for this request: claude | openai",
    )


# ─── Metrics schemas ───────────────────────────────────────────────────────────

class QueryMetricsRecord(BaseModel):
    query_id:           str
    query_text:         str
    provider:           str
    model:              str
    latency_ms:         float
    tokens_used:        Optional[int]
    estimated_cost_usd: Optional[float]
    patients_returned:  int
    timestamp:          datetime


# ─── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str = "ok"
    version:  str
    provider: str
    db:       str = "connected"
