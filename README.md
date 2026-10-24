# AI Knowledge Assistant

An AI-powered knowledge assistant that lets users upload PDF documents and ask questions about their content. Built with **FastAPI, LangChain, Qdrant, and React**, the application uses Retrieval-Augmented Generation (RAG) to retrieve relevant document context and generate grounded answers with source references.

## Features

- **PDF ingestion:** Upload PDF documents and extract their text for indexing.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks to provide context-aware answers.
- **Semantic search:** Use vector embeddings and Qdrant for similarity-based retrieval.
- **Streaming responses:** Stream generated answers to the frontend using Server-Sent Events (SSE).
- **Source references:** Display retrieved sources alongside generated answers.
- **Interactive frontend:** Chat-style interface built with React, TypeScript, and Tailwind CSS.
- **Document management:** View uploaded documents and upload additional PDFs.
- **API authentication:** Protect application endpoints with API-key authentication.
- **Rate limiting:** Apply request limits to selected API endpoints.
- **Input validation:** Validate PDF uploads, file sizes, filenames, and document readability.
- **Automated testing:** Test backend functionality using Pytest and HTTPX.
- **Docker support:** Run the backend and Qdrant locally using Docker Compose.
- **Retrieval evaluation:** Evaluate retrieval quality using Recall@3 test cases.
- **CI:** Run automated tests through GitHub Actions.

## Architecture

```text
                  User
                   |
                   v
          React + TypeScript
          Tailwind CSS + Vite
                   |
             HTTP / SSE
                   |
                   v
             FastAPI API
                   |
          +--------+--------+
          |                 |
          v                 v
    Document Ingestion   RAG Pipeline
          |                 |
          v                 v
     PDF Extraction     Embeddings
                            |
                            v
                       Qdrant DB
                            |
                            v
                    Relevant Context
                            |
                            v
                     LLM Provider
                            |
                            v
                   Answer + Sources
```

The frontend communicates with FastAPI. Uploaded PDFs are processed and indexed for retrieval. When a user asks a question, the application retrieves relevant document content and uses it as context for answer generation.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Styling | Tailwind CSS |
| Backend | Python, FastAPI |
| RAG orchestration | LangChain |
| Vector database | Qdrant |
| Embeddings | Jina AI |
| LLM provider | OpenRouter |
| API communication | HTTP, Server-Sent Events |
| Authentication | API-key authentication |
| Rate limiting | SlowAPI |
| PDF processing | pypdf |
| Testing | Pytest, HTTPX |
| Dependency management | uv |
| Containerization | Docker, Docker Compose |
| CI | GitHub Actions |

## Project Structure

```text
ai-knowledge-assistant/
├── app/
│   ├── api.py
│   ├── config.py
│   ├── ingestion.py
│   ├── rag.py
│   └── retrieval.py
├── evaluation/
│   ├── rag_cases.json
│   └── evaluate_retrieval.py
├── tests/
│   └── test_api.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── .env.local
│   ├── package.json
│   └── tailwind.config.js
├── .github/
│   └── workflows/
│       └── tests.yml
├── main.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Prerequisites

Install the following tools before running the application:

- Python 3.12
- uv
- Node.js and npm
- Docker Desktop with Docker Compose
- API credentials for the configured embedding and LLM providers

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Arshad-khan147/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Configure environment variables

Create a local `.env` file in the project root using the environment-variable names expected by `app/config.py`.

Example configuration:

```dotenv
API_KEY=replace_with_a_strong_random_key
JINA_API_KEY=your_jina_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_URL=http://localhost:6333
```

Use the exact additional variable names required by your application's configuration, including any model or collection settings.

**Security:** Never commit `.env` or real API credentials to GitHub. Store production secrets in your hosting provider's environment-variable settings. Rotate any credentials that may have been exposed.

### 3. Start the backend and Qdrant

From the repository root:

```bash
docker compose up -d --build
```

Check the running services:

```bash
docker compose ps
```

Check the API health endpoint:

```bash
curl http://localhost:8000/health
```

The API should be available at:

- API base URL: `http://localhost:8000`
- Interactive API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Qdrant dashboard: `http://localhost:6333/dashboard`

The Qdrant dashboard is available locally if exposed by your Docker Compose configuration.

### 4. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

Start the Vite development server:

```bash
npm run dev
```

Open the local URL printed by Vite, typically:

`http://localhost:5173`

If Vite selects another port, use the URL shown in the terminal.

### 5. Build the frontend

From the `frontend` directory:

```bash
npm run build
```

The production build is generated in `frontend/dist`.

## API Endpoints

The following endpoints are implemented in the application.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/ask` | Submit a question and receive an answer |
| POST | `/ask/stream` | Stream an answer using SSE |
| GET | `/documents` | List uploaded documents |
| POST | `/documents/upload` | Upload and process a PDF |

Protected endpoints require the configured API key. Check `/docs` for request schemas, response formats, and authentication requirements.

## How RAG Works

1. **Upload:** A user uploads a PDF through the frontend.
2. **Extract:** The backend extracts readable text from the document.
3. **Chunk:** The extracted text is divided into smaller pieces for retrieval.
4. **Embed:** The embedding provider converts text chunks into vector representations.
5. **Index:** Vectors and associated metadata are stored in Qdrant.
6. **Retrieve:** A user asks a question, and the application retrieves relevant chunks using similarity search.
7. **Generate:** The retrieved context is passed to the configured LLM provider.
8. **Respond:** The frontend displays the generated answer and available source references.

## Testing

Run the backend tests from the repository root:

```bash
uv run pytest
```

The project also includes retrieval evaluation cases.

Run the evaluation script:

```bash
uv run python evaluation/evaluate_retrieval.py
```

The evaluation script checks retrieval performance using Recall@3. Results depend on the current dataset, embedding configuration, and Qdrant contents.

## Docker Management

Start or rebuild services:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f
```

Stop services:

```bash
docker compose down
```

Review your Docker Compose volume configuration before removing containers or volumes. Persistent volumes may contain your indexed document data.

## Deployment

The application consists of three separately deployable components.

### Frontend

The React/Vite frontend can be deployed to Netlify or another static hosting provider connected to GitHub.

Recommended build settings:

- Base directory: `frontend`
- Build command: `npm run build`
- Publish directory: `dist`

Set the `VITE_API_BASE_URL` environment variable to the public HTTPS URL of the deployed FastAPI backend, then trigger a new frontend build.

### Backend

Deploy the existing FastAPI application using the project's Dockerfile or another compatible Python hosting configuration.

Configure production environment variables for API authentication, embeddings, the LLM provider, and Qdrant. Set CORS to allow the deployed frontend origin.

### Vector database

Configure Qdrant using a persistent hosted database or another durable deployment. Set the production Qdrant URL and API key in the backend's environment.

**Important:** Do not rely on a temporary application filesystem to preserve uploaded files or vector data. Verify data persistence, backup requirements, service limits, and free-tier restrictions before production use.

### Live Application

- Frontend: `Coming soon`
- Backend API: `Coming soon`
- API documentation: Available at `/docs` on the deployed backend

## Future Improvements

- Add user accounts and document-level access controls.
- Introduce document deletion and re-indexing.
- Improve retrieval with metadata filtering and reranking.
- Expand automated RAG evaluation with additional test cases.
- Add observability, structured logging, and error tracking.
- Configure persistent document storage and automated backups.
- Deploy the application with production-grade authentication and monitoring.

## Author

**Arshad Khan**

GitHub: [Arshad-khan147](https://github.com/Arshad-khan147)

Repository: [AI Knowledge Assistant](https://github.com/Arshad-khan147/ai-knowledge-assistant)
