#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_BE=1
START_FE=1

usage() {
  cat <<EOF
Usage: $0 [--be | --fe]
  (no flag)  start both backend and frontend
  --be       start only the backend (FastAPI on :8000)
  --fe       start only the frontend (Vite on :5173)
EOF
}

case "${1:-}" in
  --be) START_FE=0 ;;
  --fe) START_BE=0 ;;
  "")   ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
esac

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "Shutting down..."
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

if [[ "$START_BE" -eq 1 ]]; then
  echo "Starting backend on http://localhost:8000"
  (cd "$ROOT_DIR/backend" && uv run uvicorn app.main:app --reload --port 8000) &
  BACKEND_PID=$!
fi

if [[ "$START_FE" -eq 1 ]]; then
  echo "Starting frontend on http://localhost:5173"
  (cd "$ROOT_DIR/frontend" && npm run dev) &
  FRONTEND_PID=$!
fi

wait
