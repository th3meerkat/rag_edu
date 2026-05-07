"""Ingestion entrypoint for the LlamaIndex RAG service.

Usage (from backend/):
    uv run python -m tools.ingest_llamaindex          # incremental (skips PDFs in manifest)
    uv run python -m tools.ingest_llamaindex --reset  # drop collection + wipe manifest, then reingest all

Why a dedicated script: load_dotenv and logging config live here so that
`run_ingestion()` itself has no side effects on env/logging (callers differ:
FastAPI loads dotenv in main.py; CLI loads it here).
"""
import argparse
import logging
import sys

from dotenv import load_dotenv

from app.config import ENV_PATH, setup_logging
from app.services.llamaindex_rag.db_comm import reset_collection
from app.services.llamaindex_rag.service import LlamaindexSrv
from app.services.utils import manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LlamaIndex RAG ingestion.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the Chroma collection and the manifest before ingesting.",
    )
    args = parser.parse_args(argv)

    load_dotenv(ENV_PATH)
    setup_logging()
    logger = logging.getLogger("ingest_llamaindex")

    if args.reset:
        logger.info("--reset: dropping collection and manifest")
        reset_collection()
        mpath = manifest_path(LlamaindexSrv.engine_name)
        if mpath.exists():
            mpath.unlink()
            logger.info("Removed manifest at %s", mpath)

    LlamaindexSrv().run_ingestion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
