from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "app" / "data"
INGESTED_MANIFEST = DATA_DIR / "ingested.json"
ENV_PATH = BACKEND_DIR / ".env"

POSITIONAL_WINDOW_PCT = 0.10

EXPANSION_MODEL = "gpt-4o-mini"
N_EXPANDED = 2

TOP_K_PER_QUERY = 5
RRF_K = 60
TOP_K_AFTER_FUSION = 10
TOP_K_FINAL = 3

RERANKER_URL = "http://localhost:8002"
RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"