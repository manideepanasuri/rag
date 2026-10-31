import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["indexed_documents"] >= 6


def test_ask_covered_question(client):
    response = client.post(
        "/api/v1/ask",
        json={"question": "How do I reset my password?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["doc_id"] == "D-01"
    assert "reset link" in data["answer"].lower() or "forgot password" in data["answer"].lower()


def test_ask_uncovered_negative_question(client):
    response = client.post(
        "/api/v1/ask",
        json={"question": "Can I connect Solstice to Salesforce?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_covered"
    assert data["answer"] == "I don't have an answer for that."
    assert data["doc_id"] is None


def test_docs_crud_lifecycle(client):
    # 1. Create a new document
    new_doc = {
        "doc_id": "D-99",
        "title": "Custom Webhooks",
        "content": "Webhooks allow sending real-time HTTP POST notifications to third-party endpoints on course completion.",
    }
    create_resp = client.post("/api/v1/docs", json=new_doc)
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    assert created_data["doc_id"] == "D-99"

    # 2. Search for new doc content
    ask_resp = client.post(
        "/api/v1/ask",
        json={"question": "How do course completion webhooks work?"},
    )
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["status"] == "answered"
    assert ask_data["doc_id"] == "D-99"

    # 3. Delete document
    del_resp = client.delete("/api/v1/docs/D-99")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 4. Confirm 404 on deleting non-existent
    del_fail = client.delete("/api/v1/docs/D-99")
    assert del_fail.status_code == 404
