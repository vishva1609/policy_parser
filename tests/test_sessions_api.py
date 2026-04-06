"""Sessions API — contract tests (no full LLM pipeline)."""
import io

import pytest
from fastapi.testclient import TestClient

from app.fastapi_app import app

MIN_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_create_session(client):
    r = client.post(
        "/sessions",
        files={
            "policy_a": ("a.pdf", io.BytesIO(MIN_PDF), "application/pdf"),
            "policy_b": ("b.pdf", io.BytesIO(MIN_PDF), "application/pdf"),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 32


def test_session_invalid_id_format(client):
    r = client.get("/sessions/not-a-valid-uuid-hex-string-at-all")
    assert r.status_code == 400


def test_session_not_found(client):
    r = client.get("/sessions/" + "0" * 32)
    assert r.status_code == 404


def test_create_session_single_file_mode(client):
    r = client.post(
        "/sessions",
        files={
            "policy_a": ("a.pdf", io.BytesIO(MIN_PDF), "application/pdf"),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["session_mode"] == "single_policy_a"
    assert "note" in data
    gr = client.get(f"/sessions/{data['session_id']}")
    assert gr.json()["paths"]["policy_b"] is None


def test_single_file_session_compare_rejected(client):
    cr = client.post(
        "/sessions",
        files={"policy_a": ("a.pdf", io.BytesIO(MIN_PDF), "application/pdf")},
    )
    sid = cr.json()["session_id"]
    r = client.post(f"/sessions/{sid}/steps/compare")
    assert r.status_code == 400
    assert "policy A" in r.json()["detail"] or "both" in r.json()["detail"].lower()


def test_get_session_after_create(client):
    cr = client.post(
        "/sessions",
        files={
            "policy_a": ("a.pdf", io.BytesIO(MIN_PDF), "application/pdf"),
            "policy_b": ("b.pdf", io.BytesIO(MIN_PDF), "application/pdf"),
        },
    )
    sid = cr.json()["session_id"]
    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert "steps" in body
    assert body["steps"]["parse"]["done"] is False
