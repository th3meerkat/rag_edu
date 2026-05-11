"""Framework-agnostic utilities shared by every RAG engine.

Nothing here imports LangChain / LlamaIndex: each engine wraps these primitives
with its own native types (see `langchain_rag/utils.py` and
`llamaindex_rag/utils.py`).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, TypeVar

import httpx

from app.config import (
    INGESTED_DATA_DIR,
    POSITIONAL_WINDOW_PCT,
    RERANKER_MODEL,
    RERANKER_URL,
    RRF_K,
    TOP_K_AFTER_FUSION,
    TOP_K_FINAL,
)

T = TypeVar("T")

# (kind, value): "pagina"|"capitulo" llevan N; "final"|"principio" llevan None.
PositionalIntent = tuple[str, int | None]

_PAGINA_RE = re.compile(r"\bp[áa]gina\s+(\d+)\b", re.IGNORECASE)
_CAPITULO_RE = re.compile(r"\bcap[íi]tulo\s+(\d+)\b", re.IGNORECASE)
_FINAL_RE = re.compile(
    r"\b(al\s+final|final\s+del|desenlace|última\s+p[áa]gina)\b", re.IGNORECASE
)
_PRINCIPIO_RE = re.compile(
    r"\b(al\s+principio|al\s+comienzo|al\s+inicio|principio\s+del|"
    r"comienzo\s+del|inicio\s+del|primera\s+p[áa]gina)\b",
    re.IGNORECASE,
)


# ---------- Per-engine manifest I/O ----------

def manifest_path(engine: str) -> Path:
    """Path to the ingestion manifest for a given engine (one file per engine)."""
    return INGESTED_DATA_DIR / f"ingested_{engine}.json"


def load_manifest(engine: str) -> dict[str, int]:
    """Read the ingestion manifest ({source: num_pages}); return {} if missing."""
    p = manifest_path(engine)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_manifest(engine: str, manifest: dict[str, int]) -> None:
    """Persist the ingestion manifest as JSON."""
    manifest_path(engine).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


# ---------- Positional detection + Chroma filter building ----------

def detect_positional(msg: str) -> PositionalIntent | None:
    """Detect a positional pattern in the query (page/chapter N, ending, beginning)."""
    # Orden: patrones específicos (pagina/capitulo con N) antes que los genéricos.
    m = _PAGINA_RE.search(msg)
    if m:
        return ("pagina", int(m.group(1)))
    m = _CAPITULO_RE.search(msg)
    if m:
        return ("capitulo", int(m.group(1)))
    if _FINAL_RE.search(msg):
        return ("final", None)
    if _PRINCIPIO_RE.search(msg):
        return ("principio", None)
    return None


def build_filter(
    intent: PositionalIntent, manifest: dict[str, int]
) -> tuple[dict | None, dict | None]:
    """Translate a positional intent into a Chroma filter (where, where_document).

    Both `langchain-chroma` and `llama-index-vector-stores-chroma` accept the
    same raw Chroma filter dialect, so this helper stays framework-agnostic.
    """
    kind, value = intent

    if kind == "pagina":
        assert value is not None
        # El usuario piensa en 1-indexado; PyPDF guarda 0-indexado.
        return ({"page": value - 1}, None)

    if kind == "capitulo":
        assert value is not None
        # Sin metadata de capítulo: filtramos por contenido del chunk.
        return (None, {"$contains": f"Capítulo {value}"})

    if kind in ("final", "principio"):
        # Chroma exige un único operador por objeto: $gte y $lte van en
        # cláusulas separadas dentro del $and.
        clauses: list[dict] = []
        for source, n_pages in manifest.items():
            window = max(1, math.ceil(POSITIONAL_WINDOW_PCT * n_pages))
            if kind == "final":
                lo, hi = n_pages - window, n_pages - 1
            else:
                lo, hi = 0, window - 1
            clauses.append(
                {
                    "$and": [
                        {"source": source},
                        {"page": {"$gte": lo}},
                        {"page": {"$lte": hi}},
                    ]
                }
            )
        if not clauses:
            return (None, None)
        if len(clauses) == 1:
            return (clauses[0], None)
        return ({"$or": clauses}, None)

    return (None, None)


# ---------- Generic Reciprocal Rank Fusion ----------

def rrf_fuse(
    ranked_lists: list[list[T]],
    key_fn: Callable[[T], str],
    k: int = RRF_K,
    top_n: int = TOP_K_AFTER_FUSION,
) -> list[tuple[float, T]]:
    """Fuse several ranked lists into one via Reciprocal Rank Fusion.

    `key_fn` is used to deduplicate across lists (same key → same entry).
    Pass a framework-specific projection (e.g. LangChain `doc.page_content`
    or LlamaIndex `node.node.text`) from the caller.
    """
    # Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).
    # Combines multiple ranked lists into one by summing 1 / (k + rank) across
    # the lists where the same item appears. k=60 is the widely used default.
    scores: dict[str, tuple[float, T]] = {}
    for ranked in ranked_lists:
        for rank_idx, item in enumerate(ranked):
            key = key_fn(item)
            prev_score, _ = scores.get(key, (0.0, item))
            scores[key] = (prev_score + 1.0 / (k + rank_idx + 1), item)

    ordered = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return ordered[:top_n]


# ---------- Generic reranker HTTP call ----------

def rerank_texts(
    query: str, texts: list[str], top_n: int = TOP_K_FINAL
) -> list[tuple[int, float]]:
    """Rerank a list of raw text chunks via the Infinity service.

    Returns pairs `(original_index, relevance_score)` ordered by score desc.
    Callers map the indexes back to their framework-native objects.
    """
    if not texts:
        return []
    resp = httpx.post(
        f"{RERANKER_URL}/rerank",
        json={
            "query": query,
            "documents": texts,
            "model": RERANKER_MODEL,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    # Infinity response: {"results": [{"relevance_score": float, "index": int, ...}, ...]}
    results = resp.json()["results"]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top = results[:top_n]
    return [(item["index"], item["relevance_score"]) for item in top]
