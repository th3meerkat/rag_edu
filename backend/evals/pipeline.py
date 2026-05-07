"""Retrieval-only adapter over both engines, with per-stage latency.

Composes the existing public APIs of `LangchainSrv` / `LlamaindexSrv` without
touching them. Skips generation entirely — this is a retrieval evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import TOP_K_FINAL, TOP_K_PER_QUERY
from app.services.utils import build_filter, detect_positional, load_manifest


@dataclass
class StageTimings:
    prepare_s: float = 0.0
    retrieve_fuse_s: float = 0.0
    rerank_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.prepare_s + self.retrieve_fuse_s + self.rerank_s


@dataclass
class RetrievedDoc:
    """Engine-agnostic view of a retrieved chunk."""
    page: int
    source: str
    text: str
    score: float | None = None


@dataclass
class RunResult:
    engine: str
    query_id: str
    query: str
    retrieved: list[RetrievedDoc]        # post retrieve+fuse (len ≤ TOP_K_PER_QUERY or AFTER_FUSION)
    reranked: list[RetrievedDoc]         # post rerank      (len ≤ TOP_K_FINAL)
    timings: StageTimings = field(default_factory=StageTimings)


# ---------- LangChain path ----------

def _run_langchain(query: str, query_id: str) -> RunResult:
    from app.services.langchain_rag.service import LangchainSrv
    from app.services.langchain_rag.utils import rerank

    srv = LangchainSrv()
    timings = StageTimings()

    t0 = time.perf_counter()
    state = srv._prepare_state(query)
    t1 = time.perf_counter()
    timings.prepare_s = t1 - t0

    candidates = srv._retrieve_and_fuse(state)
    t2 = time.perf_counter()
    timings.retrieve_fuse_s = t2 - t1

    reranked_pairs = rerank(query, candidates, top_n=TOP_K_FINAL)
    t3 = time.perf_counter()
    timings.rerank_s = t3 - t2

    retrieved = [
        RetrievedDoc(
            page=int(d.metadata.get("page", -1)),
            source=str(d.metadata.get("source", "?")),
            text=d.page_content,
        )
        for d in candidates
    ]
    reranked = [
        RetrievedDoc(
            page=int(d.metadata.get("page", -1)),
            source=str(d.metadata.get("source", "?")),
            text=d.page_content,
            score=float(score),
        )
        for d, score in reranked_pairs
    ]
    return RunResult(
        engine="langchain",
        query_id=query_id,
        query=query,
        retrieved=retrieved,
        reranked=reranked,
        timings=timings,
    )


# ---------- LlamaIndex path ----------

def _run_llamaindex(query: str, query_id: str) -> RunResult:
    from llama_index.core.schema import QueryBundle

    from app.services.llamaindex_rag.postprocessor import InfinityRerank
    from app.services.llamaindex_rag.service import ChromaFilterRetriever

    timings = StageTimings()

    t0 = time.perf_counter()
    intent = detect_positional(query)
    where: dict | None = None
    where_doc: dict | None = None
    if intent is not None:
        manifest = load_manifest("llamaindex")
        where, where_doc = build_filter(intent, manifest)
    t1 = time.perf_counter()
    timings.prepare_s = t1 - t0

    retriever = ChromaFilterRetriever(
        similarity_top_k=TOP_K_PER_QUERY, where=where, where_document=where_doc,
    )
    candidates = retriever.retrieve(query)
    t2 = time.perf_counter()
    timings.retrieve_fuse_s = t2 - t1

    reranker = InfinityRerank(top_n=TOP_K_FINAL)
    reranked_nodes = reranker.postprocess_nodes(
        candidates, query_bundle=QueryBundle(query_str=query),
    )
    t3 = time.perf_counter()
    timings.rerank_s = t3 - t2

    retrieved = [
        RetrievedDoc(
            page=int(n.node.metadata.get("page", -1)),
            source=str(n.node.metadata.get("source", "?")),
            text=n.node.get_content(),
            score=n.score,
        )
        for n in candidates
    ]
    reranked = [
        RetrievedDoc(
            page=int(n.node.metadata.get("page", -1)),
            source=str(n.node.metadata.get("source", "?")),
            text=n.node.get_content(),
            score=n.score,
        )
        for n in reranked_nodes
    ]
    return RunResult(
        engine="llamaindex",
        query_id=query_id,
        query=query,
        retrieved=retrieved,
        reranked=reranked,
        timings=timings,
    )


# ---------- Entrypoint ----------

ENGINES = ("langchain", "llamaindex")


def run(engine: str, query: str, query_id: str) -> RunResult:
    if engine == "langchain":
        return _run_langchain(query, query_id)
    if engine == "llamaindex":
        return _run_llamaindex(query, query_id)
    raise ValueError(f"Unknown engine: {engine}")
