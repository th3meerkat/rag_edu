import logging
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BACKEND_DIR.parent / "raw_data"
INGESTED_DATA_DIR = BACKEND_DIR / "app" / "ingested_data"
ENV_PATH = BACKEND_DIR / ".env"


def setup_logging() -> None:
    """Configure root logging. Call after `load_dotenv` so `LOG_LEVEL` is read
    from the .env. Falls back to INFO if unset or unrecognized."""
    level = logging.getLevelNamesMapping().get(
        os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

POSITIONAL_WINDOW_PCT = 0.10

EXPANSION_MODEL = "gpt-4o-mini"
N_EXPANDED = 2

TOP_K_PER_QUERY = 5
RRF_K = 60
TOP_K_AFTER_FUSION = 10
TOP_K_FINAL = 3

RERANKER_URL = "http://localhost:8002"
RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"