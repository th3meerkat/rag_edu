"""Eval orchestrator. Runs both engines over the golden set and writes an HTML report.

Usage (from `backend/`):
    uv run python -m evals.run                # both engines, with judge
    uv run python -m evals.run --no-judge     # skip LLM-as-judge
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from app.config import ENV_PATH, TOP_K_FINAL, TOP_K_PER_QUERY, setup_logging
from evals import judge, metrics_gt, pipeline, report

logger = logging.getLogger("evals.run")

DATASET_PATH = Path(__file__).parent / "datasets" / "principito_golden.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics_row(run: pipeline.RunResult, relevant_pages: set[int]) -> dict:
    retrieved_pages = [d.page for d in run.retrieved]
    reranked_pages = [d.page for d in run.reranked]
    return {
        "engine": run.engine,
        "query_id": run.query_id,
        f"hit@{TOP_K_PER_QUERY}_retrieve": metrics_gt.hit_at_k(retrieved_pages, relevant_pages),
        f"recall@{TOP_K_PER_QUERY}_retrieve": metrics_gt.recall_at_k(retrieved_pages, relevant_pages),
        f"mrr@{TOP_K_PER_QUERY}_retrieve": metrics_gt.reciprocal_rank(retrieved_pages, relevant_pages),
        f"hit@{TOP_K_FINAL}_rerank": metrics_gt.hit_at_k(reranked_pages, relevant_pages),
        f"recall@{TOP_K_FINAL}_rerank": metrics_gt.recall_at_k(reranked_pages, relevant_pages),
        f"mrr@{TOP_K_FINAL}_rerank": metrics_gt.reciprocal_rank(reranked_pages, relevant_pages),
    }


def compute_latency_row(run: pipeline.RunResult) -> dict:
    t = run.timings
    return {
        "engine": run.engine,
        "query_id": run.query_id,
        "prepare_s": t.prepare_s,
        "retrieve_fuse_s": t.retrieve_fuse_s,
        "rerank_s": t.rerank_s,
        "total_s": t.total_s,
    }


def compute_judge_row(run: pipeline.RunResult) -> dict:
    """Run the LLM judge on the reranked (top-k) chunks and report the mean."""
    scores: list[int] = []
    for doc in run.reranked:
        v = judge.score_chunk(run.query, doc.text)
        if v is not None:
            scores.append(v)
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"engine": run.engine, "query_id": run.query_id, "context_relevance_mean": mean}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only RAG evaluation.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge (Ollama).")
    args = parser.parse_args(argv)

    load_dotenv(ENV_PATH)
    setup_logging()

    if not args.no_judge:
        try:
            judge.preflight()
        except judge.JudgeUnavailable as e:
            logger.error("%s", e)
            return 1
        logger.info("Preloading '%s' into Ollama...", judge.JUDGE_MODEL)
        judge.load_model()

    from app.services.langchain_rag.db_comm import get_collection_count as _lc_count
    from app.services.langchain_rag.service import LangchainSrv
    from app.services.llamaindex_rag.db_comm import get_collection_count as _li_count
    from app.services.llamaindex_rag.service import LlamaindexSrv
    from app.services.utils import manifest_path

    for name, srv_cls, fn in (
        ("langchain", LangchainSrv, _lc_count),
        ("llamaindex", LlamaindexSrv, _li_count),
    ):
        if fn() == 0:
            logger.warning(
                "Collection '%s' empty — auto-ingesting "
                "(this will call OpenAI embeddings API on the PDFs)", name,
            )
            # Drop the manifest so run_ingestion doesn't skip already-listed
            # PDFs whose embeddings were lost (e.g. collection was reset).
            mpath = manifest_path(name)
            if mpath.exists():
                mpath.unlink()
            srv_cls().run_ingestion()
            logger.info("Collection '%s' now has %d embeddings", name, fn())
        else:
            logger.info("Collection '%s': %d embeddings", name, fn())

    golden = load_golden(DATASET_PATH)
    logger.info("Loaded %d queries from %s", len(golden), DATASET_PATH)

    metrics_rows: list[dict] = []
    latency_rows: list[dict] = []
    judge_rows: list[dict] = []

    for engine in pipeline.ENGINES:
        logger.info("=== Engine: %s ===", engine)
        for item in golden:
            relevant = set(item["relevant_pages"])
            run = pipeline.run(engine, item["query"], item["id"])
            logger.info(
                "[%s] %s retrieve=%d rerank=%d prepare=%.3fs retrieve_fuse=%.3fs rerank=%.3fs",
                engine, item["id"],
                len(run.retrieved), len(run.reranked),
                run.timings.prepare_s, run.timings.retrieve_fuse_s, run.timings.rerank_s,
            )
            metrics_rows.append(compute_metrics_row(run, relevant))
            latency_rows.append(compute_latency_row(run))
            if not args.no_judge:
                judge_rows.append(compute_judge_row(run))

    metrics_df = pd.DataFrame(metrics_rows).set_index(["engine", "query_id"])
    latency_df = pd.DataFrame(latency_rows).set_index(["engine", "query_id"])
    judge_df = (
        pd.DataFrame(judge_rows).set_index(["engine", "query_id"])
        if judge_rows else None
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_path = report.build(metrics_df, latency_df, judge_df, REPORTS_DIR, timestamp)

    summary = metrics_df.groupby("engine").mean(numeric_only=True).round(4)
    logger.info("Report written to %s", html_path)
    print("\n=== SUMMARY (mean per engine) ===")
    print(summary.to_string())
    print(f"\nReport: {html_path}")

    if not args.no_judge:
        logger.info("Unloading '%s' from Ollama...", judge.JUDGE_MODEL)
        judge.stop_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
