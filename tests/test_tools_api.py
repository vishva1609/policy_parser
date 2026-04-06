"""Tools API — lightweight contract tests."""
import io

import pytest
from fastapi.testclient import TestClient

from app.fastapi_app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_tools_tuning_available_list(client):
    r = client.get("/tools/tuning/available")
    assert r.status_code == 200
    data = r.json()
    assert "tuning_files" in data
    assert isinstance(data["tuning_files"], list)


def test_semantic_search_requires_query(client):
    r = client.post(
        "/tools/semantic-search",
        files={"policy": ("p.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        data={"query": "   "},
    )
    assert r.status_code == 422


def test_questions_from_policy_503_without_api_key(client, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    from app import fastapi_app

    fresh = fastapi_app.create_fastapi_app()
    with TestClient(fresh) as c:
        r = c.post(
            "/tools/questions-from-policy",
            files={"policy": ("p.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            data={"questions_per_statement": 1},
        )
    assert r.status_code == 503
