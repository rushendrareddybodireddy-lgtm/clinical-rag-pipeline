"""
Shared test fixtures.

The integration tests require a running Postgres instance with pgvector.
Set TEST_DATABASE_URL in your environment to point at a test DB, or spin up
the docker-compose stack and use the default settings.

Unit tests are fully isolated (no DB, no LLM calls).
"""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def settings():
    from api.config import Settings
    return Settings(
        postgres_host=os.getenv("TEST_POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("TEST_POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("TEST_POSTGRES_DB", "clinical_rag_test"),
        postgres_user=os.getenv("TEST_POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("TEST_POSTGRES_PASSWORD", "postgres"),
        anthropic_api_key="test-key-not-real",
        openai_api_key="test-key-not-real",
        llm_provider="claude",
    )


@pytest.fixture
def api_client():
    """TestClient with LLM calls mocked out."""
    with patch("api.routers.alerts._fetch_alert_rows") as mock_alerts, \
         patch("api.routers.alerts._retriever") as mock_retriever, \
         patch("api.routers.alerts.get_llm_client") as mock_llm:

        mock_alerts.return_value = [
            {
                "icustay_id": 200001,
                "sofa_total": 11,
                "risk_tier": "high",
                "sofa_resp": 3, "sofa_coag": 2, "sofa_liver": 1,
                "sofa_cardio": 2, "sofa_cns": 2, "sofa_renal": 1,
                "score_window_end": "2150-01-01T12:00:00",
            }
        ]
        mock_retriever.retrieve.return_value = []

        mock_response = MagicMock()
        mock_response.text = "Patient 200001 is at elevated sepsis risk due to respiratory failure (SOFA resp=3)."
        mock_response.provider = "claude"
        mock_response.model = "claude-opus-4-6"
        mock_response.total_tokens = 512
        mock_response.estimated_cost_usd.return_value = 0.000256

        mock_llm_instance = MagicMock()
        mock_llm_instance.chat = AsyncMock(return_value=mock_response)
        mock_llm.return_value = mock_llm_instance

        from api.main import app
        with TestClient(app) as client:
            yield client
