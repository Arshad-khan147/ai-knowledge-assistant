
from fastapi.testclient import TestClient
import pytest
from io import BytesIO
from pypdf import PdfWriter
import app.api as api
from types import SimpleNamespace

import app.rag as rag
from langchain_core.documents import Document


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


def test_upload_rejects_fake_pdf(auth_headers):
    response = client.post(
        "/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "fake.pdf",
                b"%PDF-not-a-real-pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is not a readable PDF."
    )


def test_upload_rejects_oversized_file(
    auth_headers, monkeypatch
):
    monkeypatch.setattr(api, "MAX_PDF_SIZE", 10)

    response = client.post(
        "/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * 20,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"].lower()


def test_upload_rejects_encrypted_pdf(auth_headers):
    pdf_buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("test-password")
    writer.write(pdf_buffer)

    response = client.post(
        "/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "encrypted.pdf",
                pdf_buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Encrypted PDFs are not supported."
    )


def test_rag_adds_source_ids_to_context(monkeypatch):
    documents = [
        (
            Document(
                page_content="FastAPI supports asynchronous endpoints.",
                metadata={"source": "fastapi.pdf", "page": 2},
            ),
            0.85,
        ),
        (
            Document(
                page_content="Python supports async and await.",
                metadata={"source": "python.pdf", "page": 4},
            ),
            0.80,
        ),
    ]

    captured = {}

    class FakeRagChain:
        def invoke(self, payload):
            captured.update(payload)
            return SimpleNamespace(
                content="FastAPI supports asynchronous endpoints [S1]."
            )

    monkeypatch.setattr(
        rag, "retrieve_documents", lambda question, k: documents
    )
    monkeypatch.setattr(rag, "rag_chain", FakeRagChain())

    result = rag.ask_question("How does FastAPI handle async?", k=2)

    assert "[S1]" in captured["context"]
    assert "[S2]" in captured["context"]
    assert "fastapi.pdf" in captured["context"]
    assert "python.pdf" in captured["context"]

    assert result["sources"][0]["id"] == "S1"
    assert result["sources"][1]["id"] == "S2"
    assert result["sources"][0]["page"] == 2


def test_rag_skips_llm_when_no_relevant_documents(monkeypatch):
    documents = [
        (
            Document(
                page_content="Information about cooking.",
                metadata={"source": "cooking.pdf", "page": 1},
            ),
            0.30,
        ),
    ]

    class FakeRagChain:
        def invoke(self, payload):
            raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        rag, "retrieve_documents", lambda question, k: documents
    )
    monkeypatch.setattr(rag, "rag_chain", FakeRagChain())

    result = rag.ask_question("Explain database indexing", k=3)

    assert result["answer"] == rag.NO_ANSWER
    assert result["sources"] == []


def test_rag_returns_matching_source_metadata(monkeypatch):
    documents = [
        (
            Document(
                page_content="Qdrant stores and searches vectors.",
                metadata={"source": "qdrant-guide.pdf", "page": 7},
            ),
            0.91,
        ),
    ]

    class FakeRagChain:
        def invoke(self, payload):
            return SimpleNamespace(
                content="Qdrant searches vectors [S1]."
            )

    monkeypatch.setattr(
        rag, "retrieve_documents", lambda question, k: documents
    )
    monkeypatch.setattr(rag, "rag_chain", FakeRagChain())

    result = rag.ask_question("What does Qdrant do?", k=1)

    assert "[S1]" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["id"] == "S1"
    assert result["sources"][0]["source"] == "qdrant-guide.pdf"
    assert result["sources"][0]["page"] == 7
    assert result["sources"][0]["score"] == 0.91
