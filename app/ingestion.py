
from pathlib import Path
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models

from app.config import (
    JINA_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    QDRANT_URL,
    COLLECTION_NAME,
)

DOCUMENTS_DIR = Path("documents")




def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=JINA_API_KEY,
        base_url="https://api.jina.ai/v1",
        check_embedding_ctx_length=False,
    )


def load_pdf_documents():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR.resolve()}"
        )

    all_documents = []

    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")

        pages = PyPDFLoader(str(pdf_path)).load()

        for page in pages:
            page.metadata["page"] = int(page.metadata["page"]) + 1
            page.metadata["source"] = pdf_path.name

        all_documents.extend(pages)
        print(f"Extracted {len(pages)} pages.")

    return all_documents


def get_existing_fingerprints(client):
    """Read existing chunk content and metadata without downloading vectors."""
    fingerprints = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            content = payload.get("page_content", "")
            source = metadata.get("source")
            page = metadata.get("page")

            if content and source is not None and page is not None:
                fingerprints.add((source, int(page), content))

        if next_offset is None:
            break

        offset = next_offset

    return fingerprints


def ingest_documents():
    pages = load_pdf_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(pages)

    print(f"\nTotal chunks created: {len(chunks)}")

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    try:
        # Create the collection only if it doesn't already exist.
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                ),
            )
            print("Created a new Qdrant collection.")
        else:
            print("Using the existing Qdrant collection.")

        existing = get_existing_fingerprints(client)
        print(f"Existing unique chunks found: {len(existing)}")

        # Track repeated identical chunks on the same PDF page.
        occurrences = defaultdict(int)
        new_chunks = []
        new_ids = []

        for doc in chunks:
            source = doc.metadata["source"]
            page = int(doc.metadata["page"])
            content = doc.page_content

            fingerprint = (source, page, content)

            if fingerprint in existing:
                continue

            occurrence_key = (source, page, content)
            occurrence = occurrences[occurrence_key]
            occurrences[occurrence_key] += 1

            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{COLLECTION_NAME}:{source}:{page}:{occurrence}:{content}",
                )
            )

            new_chunks.append(doc)
            new_ids.append(chunk_id)

            # Prevent identical chunks in this same batch being queued twice.
            existing.add(fingerprint)

        if new_chunks:
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=get_embeddings(),
            )

            vector_store.add_documents(
                documents=new_chunks,
                ids=new_ids,
            )

            print(f"New chunks inserted: {len(new_chunks)}")
        else:
            print("No new chunks to insert. Everything is up to date.")

        info = client.get_collection(COLLECTION_NAME)
        print(f"Total points in Qdrant: {info.points_count}")
        print("PDF ingestion complete!")

    finally:
        client.close()



def ingest_pdf_file(pdf_path: Path, source_name: str):
    """Ingest one PDF without deleting existing Qdrant data."""

    pages = PyPDFLoader(str(pdf_path)).load()

    if not pages:
        raise ValueError("The PDF contains no readable pages.")

    for page in pages:
        page.metadata["page"] = int(page.metadata["page"]) + 1
        page.metadata["source"] = source_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(pages)

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                ),
            )

        existing = get_existing_fingerprints(client)

        new_chunks = []
        new_ids = []
        occurrences = {}

        for doc in chunks:
            source = doc.metadata["source"]
            page = int(doc.metadata["page"])
            content = doc.page_content
            fingerprint = (source, page, content)

            if fingerprint in existing:
                continue

            occurrence = occurrences.get(fingerprint, 0)
            occurrences[fingerprint] = occurrence + 1

            from uuid import NAMESPACE_URL, uuid5

            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{COLLECTION_NAME}:{source}:{page}:{occurrence}:{content}",
                )
            )

            new_chunks.append(doc)
            new_ids.append(chunk_id)
            existing.add(fingerprint)

        if new_chunks:
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=get_embeddings(),
            )
            vector_store.add_documents(
                documents=new_chunks,
                ids=new_ids,
            )

        return {
            "filename": source_name,
            "pages": len(pages),
            "chunks_created": len(chunks),
            "chunks_added": len(new_chunks),
            "chunks_skipped": len(chunks) - len(new_chunks),
        }

    finally:
        client.close()
