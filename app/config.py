
import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

JINA_API_KEY = os.environ["JINA_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

EMBEDDING_MODEL = "jina-embeddings-v2-base-en"
EMBEDDING_DIMENSIONS = 768

LLM_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini",
)

COLLECTION_NAME = "python_knowledge"
