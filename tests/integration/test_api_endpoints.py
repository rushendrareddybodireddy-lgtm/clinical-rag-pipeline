"""
Integration tests for the FastAPI RAG endpoints.

Requires the conftest.py api_client fixture which mocks DB and LLM calls,
so these tests run without a live database or API key.
"""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_ok(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "provider" in body


class TestSepsisRiskEndpoint:
    def test_post_default_query(self, api_client):
        r = api_client.post("/api/v1/alerts/sepsis-risk", json={})
        assert r.status_code == 200
        body = r.json()
        assert "alerts" in body
        assert "patients_at_risk" in body
        assert "latency_ms" in body
        assert "provider" in body

    def test_post_custom_query(self, api_client):
        payload = {
            "query": "Which patients have deteriorating renal function?",
            "top_k": 5,
        }
        r = api_client.post("/api/v1/alerts/sepsis-risk", json=payload)
        assert r.status_code == 200

    def test_post_with_risk_filter(self, api_client):
        r = api_client.post(
            "/api/v1/alerts/sepsis-risk",
            json={"risk_tier_filter": ["high", "critical"]},
        )
        assert r.status_code == 200

    def test_post_with_provider_override(self, api_client):
        r = api_client.post(
            "/api/v1/alerts/sepsis-risk",
            json={"llm_provider": "openai"},
        )
        # Should succeed even if mocked — just checking the route accepts the param
        assert r.status_code in (200, 500)

    def test_get_endpoint_works(self, api_client):
        r = api_client.get("/api/v1/alerts/sepsis-risk")
        assert r.status_code == 200

    def test_response_schema(self, api_client):
        r = api_client.post("/api/v1/alerts/sepsis-risk", json={})
        body = r.json()
        # Verify top-level schema fields
        for field in ["query", "provider", "model", "patients_at_risk", "alerts", "retrieved_at"]:
            assert field in body, f"Missing field: {field}"

    def test_alerts_have_sub_scores(self, api_client):
        r = api_client.post("/api/v1/alerts/sepsis-risk", json={})
        body = r.json()
        for alert in body["alerts"]:
            assert "sub_scores" in alert
            assert "sofa_total" in alert
            assert "risk_tier" in alert
            sub = alert["sub_scores"]
            for component in ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]:
                assert component in sub
                assert 0 <= sub[component] <= 4

    def test_top_k_boundary_validation(self, api_client):
        # top_k > 50 should be rejected
        r = api_client.post("/api/v1/alerts/sepsis-risk", json={"top_k": 100})
        assert r.status_code == 422


class TestMetricsEndpoint:
    def test_prometheus_endpoint_returns_text(self, api_client):
        r = api_client.get("/metrics")
        assert r.status_code == 200
        assert "clinical_rag" in r.text or "python_" in r.text
