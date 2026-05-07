"""Ingestion entrypoint for the Langchain RAG service.

Usage (from backend/):
    uv run python -m tools.ingest          # incremental (skips PDFs in manifest)
    uv run python -m tools.ingest --reset  # drop collection + wipe manifest, then reingest all

Why a dedicated script: load_dotenv and logging config live here so that
`run_ingestion()` itself has no side effects on env/logging (callers differ:
FastAPI loads dotenv in main.py; CLI loads it here).
"""
import argparse
import logging
import sys

from dotenv import load_dotenv

from app.config import ENV_PATH, setup_logging
from app.services.langchain_rag.db_comm import reset_collection
from app.services.langchain_rag.service import LangchainSrv
from app.services.utils import manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Langchain RAG ingestion.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the Chroma collection and the manifest before ingesting.",
    )
    args = parser.parse_args(argv)

    load_dotenv(ENV_PATH)
    setup_logging()
    logger = logging.getLogger("ingest")

    if args.reset:
        logger.info("--reset: dropping collection and manifest")
        reset_collection()
        mpath = manifest_path(LangchainSrv.engine_name)
        if mpath.exists():
            mpath.unlink()
            logger.info("Removed manifest at %s", mpath)

    LangchainSrv().run_ingestion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
