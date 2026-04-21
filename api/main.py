"""
Clinical RAG API — FastAPI application entry point.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or via Docker:
    docker compose up api
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.models.schemas import HealthResponse
from api.routers import alerts, metrics
from api.routers.metrics import rag_latency_histogram, rag_query_counter

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate DB connectivity on startup; clean up on shutdown."""
    log.info("Clinical RAG API starting up — provider: %s", settings.llm_provider.value)
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            dbname=settings.postgres_db, user=settings.postgres_user,
            password=settings.postgres_password,
        )
        conn.close()
        log.info("Postgres connection OK")
    except Exception as e:
        log.error("Cannot connect to Postgres on startup: %s", e)
        # Don't crash — Postgres may still be initialising in docker-compose

    yield

    log.info("Clinical RAG API shutting down.")


app = FastAPI(
    title="Clinical RAG API",
    description=(
        "LLM-powered clinical decision support for ICU sepsis risk assessment. "
        "Ingests MIMIC-III patient data through a medallion pipeline, embeds "
        "clinical notes into pgvector, and serves RAG-augmented sepsis alerts "
        "via Claude or OpenAI."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    """Record per-request latency and increment counters for Prometheus."""
    t_start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - t_start

    # Only instrument the main alert endpoint
    if "/alerts/sepsis-risk" in request.url.path:
        provider = settings.llm_provider.value
        rag_latency_histogram.labels(provider=provider).observe(latency)
        rag_query_counter.labels(provider=provider, risk_tier="all").inc()

    return response


# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(alerts.router)
app.include_router(metrics.router)


# ─── Core routes ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Liveness probe — check DB connectivity and return service info."""
    db_status = "connected"
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            dbname=settings.postgres_db, user=settings.postgres_user,
            password=settings.postgres_password,
        )
        conn.close()
    except Exception:
        db_status = "unreachable"

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=settings.llm_provider.value,
        db=db_status,
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        {"message": "Clinical RAG API", "docs": "/docs", "health": "/health"}
    )
