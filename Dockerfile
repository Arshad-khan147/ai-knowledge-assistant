
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install only application dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source, not local secrets or virtual environments.
COPY app/ ./app/

# Persistent document storage is mounted by Compose.
RUN mkdir -p /app/documents

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
