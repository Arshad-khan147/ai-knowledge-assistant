
from fastapi.testclient import TestClient
import pytest

import app.api as api


TEST_API_KEY = "test-only-secret-key"


@pytest.fixture(autouse=True)
def configure_test_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_ACCESS_KEY", TEST_API_KEY)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


client = TestClient(api.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_returns_answer_and_sources(monkeypatch, auth_headers):
    def fake_ask_question(question, k=3):
        assert question == "What is cloud computing?"
        assert k == 3

        return {
            "answer": "Cloud computing delivers computing services.",
            "sources": [
                {
                    "text": "Cloud computing provides computing services.",
                    "source": "unit 2 cloud computing.pdf",
                    "page": 2,
                    "score": 0.85,
                }
            ],
        }

    monkeypatch.setattr(api, "ask_question", fake_ask_question)

    response = client.post(
        "/ask",
        json={
            "question": "What is cloud computing?",
            "top_k": 3,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()
    assert "Cloud computing" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == "unit 2 cloud computing.pdf"


def test_ask_rejects_empty_question(auth_headers):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 3,
        },
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(auth_headers):
    response = client.post(
        "/ask",
        json={
            "question": "What is cloud computing?",
            "top_k": 0,
        },
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf(monkeypatch, tmp_path, auth_headers):
    monkeypatch.setattr(api, "DOCUMENTS_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_upload_rejects_duplicate_filename(
    monkeypatch, tmp_path, auth_headers
):
    monkeypatch.setattr(api, "DOCUMENTS_DIR", tmp_path)

    (tmp_path / "existing.pdf").write_bytes(b"%PDF-existing")

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "existing.pdf",
                b"%PDF-new-content",
                "application/pdf",
            )
        },
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_ask_rejects_missing_api_key():
    response = client.post(
        "/ask",
        json={
            "question": "What is cloud computing?",
            "top_k": 3,
        },
    )

    assert response.status_code == 401


def test_documents_requires_api_key():
    response = client.get("/documents")

    assert response.status_code == 401


def test_ask_rejects_invalid_api_key():
    response = client.post(
        "/ask",
        json={
            "question": "What is cloud computing?",
            "top_k": 3,
        },
        headers={"X-API-Key": "wrong-test-key"},
    )

    assert response.status_code == 401
