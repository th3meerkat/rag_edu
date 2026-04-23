#!/usr/bin/env bash
# Run the retrieval-only eval: both engines over the golden set, HTML report out.
#
# The script fully self-bootstraps — the user should only need to run this file:
#   - Starts the Ollama daemon in background if it is not already alive.
#   - Verifies that `gemma2:2b` is installed (else exits with the pull command).
#   - Delegates to `evals.run`, which auto-ingests missing collections, preloads
#     the model before measuring, and unloads it at the end (option B:
#     daemon stays alive, model's RAM is released).
#   - If this script started the daemon, it also kills it on exit.
#
# Flags (forwarded to `python -m evals.run`):
#   --no-judge    Skip the LLM-as-judge (and the Ollama bootstrap entirely).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/backend"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
JUDGE_MODEL="gemma2:2b"

NEED_OLLAMA=1
for arg in "$@"; do
  [[ "${arg}" == "--no-judge" ]] && NEED_OLLAMA=0
done

WE_STARTED_OLLAMA=0
OLLAMA_PID=""

is_ollama_up() {
  curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

start_ollama() {
  echo "Ollama daemon no está activo, lo arranco en background..."
  ollama serve >/tmp/ollama-evals.log 2>&1 &
  OLLAMA_PID=$!
  WE_STARTED_OLLAMA=1
  for _ in $(seq 1 20); do
    sleep 0.5
    if is_ollama_up; then
      echo "  -> listo (pid=${OLLAMA_PID})"
      return 0
    fi
  done
  echo "ERROR: Ollama no respondió tras 10s (ver /tmp/ollama-evals.log)" >&2
  return 1
}

cleanup() {
  if [[ "${WE_STARTED_OLLAMA}" -eq 1 && -n "${OLLAMA_PID}" ]]; then
    echo "Apagando el daemon Ollama que arrancó este script (pid=${OLLAMA_PID})..."
    kill "${OLLAMA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${NEED_OLLAMA}" -eq 1 ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    cat <<EOF >&2
ERROR: no se encuentra el CLI 'ollama' en PATH.

Instalá Ollama (https://ollama.com/download) o pasá --no-judge para
saltar el LLM-as-judge.
EOF
    exit 1
  fi

  if ! is_ollama_up; then
    start_ollama
  fi

  if ! ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "${JUDGE_MODEL}"; then
    cat <<EOF >&2
ERROR: el modelo '${JUDGE_MODEL}' no está descargado en este sistema.

Descargalo en otra terminal con:

    ollama pull ${JUDGE_MODEL}

y volvé a ejecutar este script.
EOF
    exit 1
  fi
fi

uv run python -m evals.run "$@"

LATEST="$(ls -t evals/reports/eval_*.html 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST}" && "$(uname)" == "Darwin" ]]; then
  open "${LATEST}" || true
fi
