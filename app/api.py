
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import secrets
import os
from fastapi import Depends, Header
from app.rag import ask_question

import json

from fastapi.responses import StreamingResponse

from app.rag import rag_chain
from app.retrieval import retrieve_documents

from pathlib import Path
from fastapi import File, UploadFile
from app.ingestion import ingest_pdf_file



app = FastAPI(
    title="PDF Knowledge Assistant API",
    description="Ask questions about your ingested PDF documents.",
    version="1.0.0",
)

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not API_ACCESS_KEY:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        API_ACCESS_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Your question about the PDF documents",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve",
    )


class SourceResponse(BaseModel):
    text: str
    source: str
    page: int | str
    score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post(
    "/ask",
    response_model=QuestionResponse,
    dependencies=[Depends(require_api_key)],
)
def ask(request: QuestionRequest):
    try:
        result = ask_question(
            question=request.question,
            k=request.top_k,
        )
        return result
    except Exception as exc:
        # Log the full exception on the server; avoid exposing secrets
        # or internal error details to API clients.
        print(f"RAG request failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question. Check the server logs.",
        ) from exc


@app.post(
    "/ask/stream",
    dependencies=[Depends(require_api_key)],
)
def ask_stream(request: QuestionRequest):
    def event_stream():
        try:
            # 1. Retrieve relevant document chunks.
            results = retrieve_documents(
                question=request.question,
                k=request.top_k,
            )

            # 2. Apply the same relevance threshold.
            threshold = 0.70
            relevant_results = [
                (document, score)
                for document, score in results
                if float(score) >= threshold
            ]

            # 3. If nothing is relevant, don't call the LLM.
            if not relevant_results:
                fallback = (
                    "I couldn't find sufficiently relevant information "
                    "in the available documents to answer that question."
                )

                yield "data: " + json.dumps({
                    "type": "sources",
                    "sources": [],
                }) + "\n\n"

                yield "data: " + json.dumps({
                    "type": "token",
                    "content": fallback,
                }) + "\n\n"

                yield "data: " + json.dumps({
                    "type": "done",
                }) + "\n\n"
                return

            # 4. Prepare the context and source metadata.
            context = "\n\n".join(
                document.page_content
                for document, score in relevant_results
            )

            sources = [
                {
                    "source": document.metadata.get("source", "unknown"),
                    "page": document.metadata.get("page", "unknown"),
                    "score": round(float(score), 4),
                    "text": document.page_content,
                }
                for document, score in relevant_results
            ]

            yield "data: " + json.dumps({
                "type": "sources",
                "sources": sources,
            }) + "\n\n"

            # 5. Stream the LLM output as it arrives.
            for chunk in rag_chain.stream({
                "context": context,
                "question": request.question,
            }):
                content = chunk.content

                if isinstance(content, str) and content:
                    yield "data: " + json.dumps({
                        "type": "token",
                        "content": content,
                    }) + "\n\n"

            # 6. Signal that the stream has finished.
            yield "data: " + json.dumps({
                "type": "done",
            }) + "\n\n"

        except Exception as exc:
            # Keep internal errors out of the client response.
            print(f"Streaming request failed: {exc}")

            yield "data: " + json.dumps({
                "type": "error",
                "message": "Failed to process the question.",
            }) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



DOCUMENTS_DIR = Path("documents")
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB


@app.post(
    "/documents/upload",
    dependencies=[Depends(require_api_key)],
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    # Keep only the filename, not any user-supplied directory path.
    filename = Path(file.filename).name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists.",
        )

    content = await file.read(MAX_PDF_SIZE + 1)

    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 20 MB upload limit.",
        )

    # Basic file signature check. The PDF loader must still parse the file.
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a valid PDF signature.",
        )

    destination.write_bytes(content)

    try:
        result = ingest_pdf_file(
            pdf_path=destination,
            source_name=filename,
        )
    except Exception as exc:
        print(f"PDF ingestion failed for {filename}: {exc}")
        # Keep the file so ingestion can be retried and investigated.
        raise HTTPException(
            status_code=500,
            detail=(
                "The PDF was saved, but ingestion failed. "
                "Check the server logs before retrying."
            ),
        ) from exc
    finally:
        await file.close()

    return {
        "message": "PDF uploaded and ingested successfully.",
        **result,
    }


@app.get(
    "/documents",
    dependencies=[Depends(require_api_key)],
)
def list_documents():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        documents.append({
            "filename": pdf_path.name,
            "size_bytes": pdf_path.stat().st_size,
        })

    return {
        "count": len(documents),
        "documents": documents,
    }
