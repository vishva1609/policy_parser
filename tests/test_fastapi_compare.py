"""FastAPI comparison API — contract tests (no full pipeline / no API key required)."""
import io

import pytest

from app.fastapi_app import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_ready_shape(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert "ready" in data and isinstance(data["ready"], bool)
    assert data["llm"] in ("configured", "not_configured")


def test_compare_policies_rejects_non_pdf(client):
    r = client.post(
        "/compare/policies",
        files={
            "policy_a": ("a.txt", io.BytesIO(b"x"), "text/plain"),
            "policy_b": ("b.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf"),
        },
    )
    assert r.status_code == 422
    assert "pdf" in r.json()["detail"].lower()


def test_compare_policies_503_or_success_without_key(client, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    from app import fastapi_app

    fresh = fastapi_app.create_fastapi_app()
    with TestClient(fresh) as c:
        r = c.post(
            "/compare/policies",
            files={
                "policy_a": ("a.pdf", io.BytesIO(b"%PDF-1.4 minimal"), "application/pdf"),
                "policy_b": ("b.pdf", io.BytesIO(b"%PDF-1.4 minimal"), "application/pdf"),
            },
        )
    assert r.status_code == 503
    assert "detail" in r.json()
