#!/usr/bin/env bash
# Run the regression test suite with coverage on the services modules.
#
# Preconditions:
#   - ChromaDB running on localhost:8001 (docker compose up -d chromadb)
#   - Reranker running on localhost:8002 (docker compose up -d reranker)
#   - backend/.env with OPENAI_API_KEY
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BACKEND_DIR}"

uv run pytest tests/ \
  --cov=app.services.utils \
  --cov=app.services.rag \
  --cov=app.services.langchain_rag \
  --cov-report=term-missing \
  "$@"
