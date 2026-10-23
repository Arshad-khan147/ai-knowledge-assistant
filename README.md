# AI Knowledge Assistant — RAG API

A Dockerized Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about their content. The application retrieves relevant document passages using semantic search and uses an LLM to generate context-grounded answers.

## Overview

This project demonstrates how to build an AI-powered backend using Python, FastAPI, LangChain, vector search, and Docker.

**Key capabilities**
- PDF document upload and ingestion
- Text chunking and embedding generation
- Semantic search with Qdrant
- Question answering using retrieved document context
- Streaming answers through Server-Sent Events (SSE)
- Source metadata in responses
- Docker Compose deployment for the API and vector database
- Automated API tests using Pytest and HTTPX

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Application development |
| FastAPI | REST API |
| LangChain | RAG pipeline orchestration |
| Jina Embeddings | Text embeddings |
| Qdrant | Vector database and similarity search |
| OpenRouter | LLM API access |
| Docker Compose | Containerized services |
| uv | Python dependency management |
| Pytest and HTTPX | Automated testing |

## Architecture

```text
                 User / API Client
                        |
                        v
                  FastAPI Backend
                        |
            +-----------+-----------+
            |                       |
            v                       v
       PDF Upload              User Question
            |                       |
            v                       v
      Text Extraction         Jina Embeddings
            |                       |
            v                       v
      Text Chunking          Qdrant Similarity Search
            |                       |
            v                       v
      Jina Embeddings          Relevant Passages
            |                       |
            v                       v
           Qdrant ----------> RAG Prompt
                                    |
                                    v
                              OpenRouter LLM
                                    |
                                    v
                         Answer + Source Metadata
```

## Project Structure

```text
GenAI/
├── app/
│   ├── api.py
│   ├── config.py
│   ├── ingestion.py
│   ├── rag.py
│   └── retrieval.py
├── documents/
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

Install the following before running the application:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Jina API key
- OpenRouter API key

## Environment Configuration

Create a local `.env` file in the project root:

```dotenv
JINA_API_KEY=your_jina_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
QDRANT_URL=http://localhost:6333
```

Replace the example values with your own credentials.

**Security:** Never commit `.env` or publish API keys. The `.env` file should be excluded through `.gitignore`.

When using Docker Compose, the API connects to Qdrant using the internal service address `http://qdrant:6333`, configured in the Compose environment.

## Run with Docker Compose

Clone the repository:

```bash
git clone https://github.com/Arshad-khan147/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

Create your `.env` file as described above, then build and start the services:

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

View API logs:

```bash
docker compose logs -f api
```

Open the interactive API documentation:

**http://localhost:8000/docs**

Stop the services:

```bash
docker compose down
```

Qdrant data is stored in a Docker named volume. Stopping the services with `docker compose down` preserves that volume; deleting volumes can remove stored vector data.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| GET | `/documents` | List available documents |
| POST | `/documents/upload` | Upload and ingest a PDF |
| POST | `/ask` | Ask a question and receive a JSON response |
| POST | `/ask/stream` | Receive a streaming answer using SSE |

Use Swagger UI at `/docs` to inspect the request schemas and test the endpoints interactively.

## Running Tests

Install the project dependencies and run the tests locally:

```bash
uv sync
uv run python -m pytest -v
```

The automated API tests cover health checks, question validation, mocked question-answering responses, and upload validation.

## Current Status

- [x] FastAPI application and API endpoints
- [x] PDF ingestion and vector search integration
- [x] RAG question-answering pipeline
- [x] Docker Compose setup
- [x] Initial automated API tests
- [ ] Authentication and rate limiting
- [ ] Expanded RAG evaluation and regression tests
- [ ] CI/CD using GitHub Actions
- [ ] Production cloud deployment and monitoring

## Future Improvements

- Add user authentication and authorization.
- Improve retrieval quality with metadata filters and retrieval evaluation.
- Add stronger upload validation and configurable resource limits.
- Measure answer quality, relevance, and unsupported-answer rates.
- Automate testing and Docker builds with GitHub Actions.
- Add a React frontend with streaming chat and source citations.
- Deploy with secure secrets management, persistent storage, monitoring, and backups.

## Learning Outcomes

This project provides practical experience with:

- Designing Python REST APIs
- Implementing a RAG pipeline
- Generating embeddings and using vector databases
- Integrating external LLM APIs
- Handling streaming responses
- Containerizing multi-service applications
- Writing automated tests
- Managing application configuration and secrets

## Author

**Arshad Khan**

Python Backend Development | Generative AI | RAG Applications

GitHub: [Arshad-khan147](https://github.com/Arshad-khan147)
