
from fastapi.testclient import TestClient

import app.api as api


client = TestClient(api.app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_returns_answer_and_sources(monkeypatch):
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
    )

    assert response.status_code == 200

    body = response.json()
    assert "Cloud computing" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == "unit 2 cloud computing.pdf"


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "What is cloud computing?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf(monkeypatch, tmp_path):
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
    )

    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_upload_rejects_duplicate_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DOCUMENTS_DIR", tmp_path)

    # Simulate a document already present on disk.
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
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
