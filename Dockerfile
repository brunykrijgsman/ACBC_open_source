FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (maximises layer cache — only rebuilds if lock file changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY . .

EXPOSE 8000

# Use exec form for proper signal handling (SIGTERM reaches uvicorn)
CMD [".venv/bin/python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--config", "/app/configs/production.yaml", "--output-dir", "/data"]
