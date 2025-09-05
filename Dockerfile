# Multi-stage Dockerfile for Climate Policy Radar MCP Server
FROM python:3.11-slim AS base

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --upgrade pip uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* /app/

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY server /app/server

# Create cache directory
RUN mkdir -p /data/cache

# Set up volume for cache
VOLUME ["/data/cache"]

# Environment variables for configuration
ENV DATA_CACHE_DIR=/data/cache \
    HF_HOME=/data/hf \
    SPARQL_ENDPOINT=https://climatepolicyradar.wikibase.cloud/query/sparql \
    DATASET_REPO=ClimatePolicyRadar/all-document-text-data \
    DATASET_REVISION=main \
    PORT=8000 \
    TRANSPORT=http

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uv", "run", "python", "-m", "server.server"]