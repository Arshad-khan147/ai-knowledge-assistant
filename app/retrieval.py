
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import QDRANT_URL, COLLECTION_NAME
from app.ingestion import get_embeddings


def get_vector_store():
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60,
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError(
            "Knowledge collection not found. "
            "Run ingestion first."
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def retrieve_documents(question: str, k: int = 3):
    vector_store = get_vector_store()

    try:
        return vector_store.similarity_search_with_score(
            question,
            k=k,
        )
    finally:
        vector_store.client.close()
