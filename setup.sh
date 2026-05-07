#!/usr/bin/env bash
# One-shot setup for the rag_chat project. Idempotent — re-runnable anytime.
#
# Stages (in order):
#   1. Pre-flight: verify required binaries; print install hints if missing.
#   2. .env template: create backend/.env from a placeholder if absent.
#   3. Backend deps: uv sync (creates .venv, installs from uv.lock).
#   4. Frontend deps: npm install in frontend/.
#   5. Services: docker compose up -d, wait for chroma + reranker health.
#   6. Health check: probe chroma, reranker, backend imports.
#   7. Final banner: success or warn loudly if the OpenAI key is missing.
#
# Optional flags forwarded to backend ingest:
#   --ingest <pdf>   Run an initial ingestion of the given PDF after setup.
#                    Default behavior is no ingestion; the user can call this
#                    later (or run it via the existing tools/ingest scripts).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------- color + logging helpers ----------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  C_RESET="$(tput sgr0)"; C_BOLD="$(tput bold)"
  C_RED="$(tput setaf 1)"; C_GREEN="$(tput setaf 2)"
  C_YELLOW="$(tput setaf 3)"; C_BLUE="$(tput setaf 4)"
  C_MAGENTA="$(tput setaf 5)"; C_CYAN="$(tput setaf 6)"
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_MAGENTA=""; C_CYAN=""
fi

info()  { printf "%s[INFO]%s  %s\n"  "$C_BLUE"  "$C_RESET" "$*"; }
ok()    { printf "%s[ OK ]%s  %s\n"  "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf "%s[WARN]%s  %s\n"  "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf "%s[FAIL]%s  %s\n"  "$C_RED"   "$C_RESET" "$*" >&2; exit 1; }
step()  { printf "\n%s== %s ==%s\n"  "$C_BOLD"  "$*" "$C_RESET"; }

# ---------- arg parsing ----------
INGEST_PDF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ingest) INGEST_PDF="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--ingest <pdf>]

Sets up backend (uv), frontend (npm), and services (docker compose).
With --ingest <pdf>, also runs an initial ingestion after setup.
EOF
      exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

# ---------- OS detection (for hints only) ----------
OS="$(uname -s)"
case "$OS" in
  Darwin) OS_NAME="macOS" ;;
  Linux)  OS_NAME="Linux" ;;
  *)      fail "Unsupported OS: $OS (only macOS and Linux are supported)" ;;
esac
info "Detected OS: $OS_NAME"

install_hint() {
  # $1 = binary name
  case "$1" in
    python3) [[ "$OS" == "Darwin" ]] && echo "  brew install python@3.13" || echo "  apt install python3.13 python3.13-venv" ;;
    node|npm) [[ "$OS" == "Darwin" ]] && echo "  brew install node" || echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs" ;;
    docker)  [[ "$OS" == "Darwin" ]] && echo "  brew install --cask docker" || echo "  https://docs.docker.com/engine/install/" ;;
    uv)      echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
    curl)    [[ "$OS" == "Darwin" ]] && echo "  brew install curl" || echo "  apt install curl" ;;
    *)       echo "  (install $1 via your package manager)" ;;
  esac
}

check_bin() {
  # $1 = binary, $2 = (optional) human-readable name
  local bin="$1"
  local label="${2:-$1}"
  if ! command -v "$bin" >/dev/null 2>&1; then
    fail "Missing required binary: $label
$(install_hint "$bin")"
  fi
}

# ---------- 1. Pre-flight ----------
step "1. Pre-flight checks"
check_bin python3
check_bin node
check_bin npm
check_bin docker
check_bin curl

if ! docker compose version >/dev/null 2>&1; then
  fail "'docker compose' subcommand not available. Install Docker Desktop or the compose plugin."
fi

if ! command -v uv >/dev/null 2>&1; then
  warn "'uv' not found — installing via the official one-liner..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installer adds to ~/.cargo/bin or ~/.local/bin; try both.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv install completed but 'uv' is not on PATH. Open a new shell and re-run."
fi
ok "All required binaries present (python3 / node / npm / docker / curl / uv)."

# ---------- 2. .env template ----------
step "2. Backend .env template"
ENV_FILE="$ROOT_DIR/backend/.env"
ENV_HAS_KEY=0  # 1 if a non-placeholder OPENAI_API_KEY appears to be set

if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<'EOF'
# rag_chat backend env
# Paste your real OpenAI key below (replace the placeholder).
OPENAI_API_KEY=sk-REPLACE_ME

# Optional. One of: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO
EOF
  ok "Created backend/.env with placeholder values."
else
  ok "backend/.env already exists — not overwriting."
fi

# Detect whether the file currently has a non-placeholder key.
# Tolerates spacing around '=' and key prefixes like sk-, sk-proj-, etc.
KEY_VALUE="$(grep -E '^[[:space:]]*OPENAI_API_KEY[[:space:]]*=' "$ENV_FILE" | head -1 | sed -E 's/^[[:space:]]*OPENAI_API_KEY[[:space:]]*=[[:space:]]*//; s/^["'\'']//; s/["'\'']$//; s/[[:space:]]+$//')"
if [[ -n "$KEY_VALUE" && "$KEY_VALUE" != "sk-REPLACE_ME" ]]; then
  ENV_HAS_KEY=1
fi

# ---------- 3. Backend deps (uv) ----------
step "3. Backend dependencies (uv sync)"
if [[ -d "$ROOT_DIR/backend/.venv" ]]; then
  warn "backend/.venv exists — uv sync will reuse it (idempotent)."
fi
(cd "$ROOT_DIR/backend" && uv sync)
ok "Backend deps in sync with uv.lock."

# ---------- 4. Frontend deps (npm) ----------
step "4. Frontend dependencies (npm install)"
if [[ -d "$ROOT_DIR/frontend/node_modules" ]]; then
  warn "frontend/node_modules exists — npm install will reconcile (idempotent)."
fi
(cd "$ROOT_DIR/frontend" && npm install)
ok "Frontend deps installed."

# ---------- 5. Docker services ----------
step "5. Docker services (chromadb + reranker)"
(cd "$ROOT_DIR" && docker compose up -d)
ok "docker compose up -d issued."

wait_for_url() {
  # $1 = url, $2 = label, $3 = timeout seconds (default 120)
  local url="$1" label="$2" timeout="${3:-120}" elapsed=0
  printf "%s[WAIT]%s  %s " "$C_CYAN" "$C_RESET" "$label"
  until curl -fsS -o /dev/null "$url"; do
    sleep 2
    elapsed=$((elapsed + 2))
    printf "."
    if (( elapsed >= timeout )); then
      printf "\n"
      fail "Timed out after ${timeout}s waiting for $label ($url)"
    fi
  done
  printf "  ready (${elapsed}s)\n"
}

wait_for_url "http://localhost:8001/api/v2/heartbeat" "chromadb @ :8001" 120
wait_for_url "http://localhost:8002/health"           "reranker @ :8002" 180

# ---------- 6. Health check ----------
step "6. Health check"
if (cd "$ROOT_DIR/backend" && uv run python -c "import fastapi, chromadb, langchain, llama_index" >/dev/null 2>&1); then
  ok "Python imports OK (fastapi / chromadb / langchain / llama_index)."
else
  fail "Python import smoke test failed. Try: cd backend && uv sync"
fi

# ---------- Optional: --ingest ----------
if [[ -n "$INGEST_PDF" ]]; then
  step "Optional: initial ingestion"
  if [[ ! -f "$INGEST_PDF" ]]; then
    fail "PDF not found: $INGEST_PDF"
  fi
  if [[ "$ENV_HAS_KEY" -eq 0 ]]; then
    warn "Skipping ingestion: OPENAI_API_KEY is not set (or still the placeholder)."
  else
    info "Ingesting $INGEST_PDF (LangChain + LlamaIndex engines)..."
    (cd "$ROOT_DIR/backend" && uv run python -m tools.ingest "$INGEST_PDF")
    (cd "$ROOT_DIR/backend" && uv run python -m tools.ingest_llamaindex "$INGEST_PDF")
    ok "Ingestion completed."
  fi
fi

# ---------- 7. Final banner ----------
echo
if [[ "$ENV_HAS_KEY" -eq 1 ]]; then
  printf "%s%s" "$C_BOLD" "$C_GREEN"
  cat <<'EOF'
╔══════════════════════════════════════════════════════════════════╗
║  ✓ Setup complete. Everything is ready.                          ║
║                                                                  ║
║  Next step:    ./start.sh                                        ║
║                                                                  ║
║    backend  →  http://localhost:8000                             ║
║    frontend →  http://localhost:5173                             ║
║    chromadb →  http://localhost:8001                             ║
║    reranker →  http://localhost:8002                             ║
╚══════════════════════════════════════════════════════════════════╝
EOF
  printf "%s" "$C_RESET"
else
  printf "%s%s" "$C_BOLD" "$C_YELLOW"
  cat <<'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⚠   ACTION REQUIRED — OpenAI API key missing                   ║
║                                                                  ║
║   The infrastructure is set up, BUT the backend will fail at     ║
║   runtime because OPENAI_API_KEY is still the placeholder.       ║
║                                                                  ║
║   1. Open  backend/.env                                          ║
║   2. Replace  sk-REPLACE_ME  with your real OpenAI key           ║
║   3. Run     ./start.sh                                          ║
║                                                                  ║
║   You don't need to re-run setup.sh — only the file edit.        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
  printf "%s" "$C_RESET"
fi
echo
