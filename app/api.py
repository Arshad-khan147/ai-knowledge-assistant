
import json
import logging
import os
import secrets
from io import BytesIO
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.ingestion import ingest_pdf_file
from app.rag import ask_question, rag_chain
from app.retrieval import retrieve_documents


logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Knowledge Assistant API",
    description="Ask questions about your ingested PDF documents.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")

DOCUMENTS_DIR = Path("documents")
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB
RELEVANCE_THRESHOLD = 0.70


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
    id: str | None = None
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
@limiter.limit("10/minute")
def ask(request: Request, payload: QuestionRequest):
    try:
        return ask_question(
            question=payload.question,
            k=payload.top_k,
        )
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question. Check the server logs.",
        ) from exc


@app.post(
    "/ask/stream",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("5/minute")
def ask_stream(request: Request, payload: QuestionRequest):
    def event_stream():
        try:
            results = retrieve_documents(
                question=payload.question,
                k=payload.top_k,
            )

            relevant_results = [
                (document, float(score))
                for document, score in results
                if float(score) >= RELEVANCE_THRESHOLD
            ]

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

            context_parts = []
            sources = []

            for index, (document, score) in enumerate(
                relevant_results,
                start=1,
            ):
                source_id = f"S{index}"
                source_name = document.metadata.get("source", "unknown")
                page = document.metadata.get("page", "unknown")

                context_parts.append(
                    f"[{source_id}]\n"
                    f"Document: {source_name}\n"
                    f"Page: {page}\n"
                    f"Content:\n{document.page_content}"
                )

                sources.append({
                    "id": source_id,
                    "source": source_name,
                    "page": page,
                    "score": round(score, 4),
                    "text": document.page_content,
                })

            context = "\n\n---\n\n".join(context_parts)

            yield "data: " + json.dumps({
                "type": "sources",
                "sources": sources,
            }) + "\n\n"

            for chunk in rag_chain.stream({
                "context": context,
                "question": payload.question,
            }):
                content = chunk.content

                if isinstance(content, str) and content:
                    yield "data: " + json.dumps({
                        "type": "token",
                        "content": content,
                    }) + "\n\n"

            yield "data: " + json.dumps({
                "type": "done",
            }) + "\n\n"

        except Exception:
            logger.exception("Streaming request failed")
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


def validate_pdf_content(content: bytes) -> None:
    """Validate that uploaded bytes can be parsed as a readable PDF."""

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not have a valid PDF signature.",
        )

    try:
        reader = PdfReader(BytesIO(content), strict=False)

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported.",
            )

        if len(reader.pages) == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no readable pages.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable PDF.",
        ) from exc


@app.post(
    "/documents/upload",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

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

        validate_pdf_content(content)
        destination.write_bytes(content)

        try:
            result = ingest_pdf_file(
                pdf_path=destination,
                source_name=filename,
            )
        except Exception as exc:
            logger.exception("PDF ingestion failed for %s", filename)
            raise HTTPException(
                status_code=500,
                detail=(
                    "The PDF was saved, but ingestion failed. "
                    "Check the server logs before retrying."
                ),
            ) from exc

        return {
            "message": "PDF uploaded and ingested successfully.",
            **result,
        }

    finally:
        await file.close()


@app.get(
    "/documents",
    dependencies=[Depends(require_api_key)],
)
def list_documents():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = [
        {
            "filename": pdf_path.name,
            "size_bytes": pdf_path.stat().st_size,
        }
        for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf"))
    ]

    return {
        "count": len(documents),
        "documents": documents,
    }
