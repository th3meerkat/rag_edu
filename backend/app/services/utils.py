import json
import math
import re
from pathlib import Path

import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import EXPANSION_MODEL, INGESTED_MANIFEST, N_EXPANDED, POSITIONAL_WINDOW_PCT, RERANKER_MODEL, RERANKER_URL, RRF_K, TOP_K_AFTER_FUSION, TOP_K_FINAL


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


def load_manifest() -> dict[str, int]:
    """Read the ingestion manifest ({source: num_pages}); return {} if missing."""
    if INGESTED_MANIFEST.exists():
        return json.loads(INGESTED_MANIFEST.read_text())
    return {}


def save_manifest(manifest: dict[str, int]) -> None:
    """Persist the ingestion manifest as JSON."""
    INGESTED_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


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
    """Translate a positional intent into a Chroma filter (where, where_document)."""
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


def expand_queries(msg: str) -> list[str]:
    """Generate N paraphrases of the query via LLM to improve retrieval."""
    llm = ChatOpenAI(
        model=EXPANSION_MODEL,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    system = (
        f"Eres un asistente que reescribe consultas para mejorar la recuperación. "
        f"Genera exactamente {N_EXPANDED} consultas alternativas (paráfrasis o ampliaciones "
        f"con sinónimos o enfoques distintos) que preserven el idioma de la consulta del usuario. "
        f'Devuelve ESTRICTAMENTE un JSON con la forma: {{"queries": ["q1", "q2"]}}. Sin texto extra.'
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    data = json.loads(str(response.content))
    queries = data.get("queries", [])
    if len(queries) < N_EXPANDED:
        raise ValueError(f"Expansion returned {len(queries)} queries, expected {N_EXPANDED}")
    return queries[:N_EXPANDED]


def rrf_fuse(
    ranked_lists: list[list[Document]], k: int = RRF_K, top_n: int = TOP_K_AFTER_FUSION
) -> list[tuple[float, Document]]:
    """Fuse several ranked lists into one via Reciprocal Rank Fusion."""
    # Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).
    # Combines multiple ranked lists into one by summing 1 / (k + rank) across
    # the lists where the same document appears. k=60 is the widely used default.
    # Deduplication key: page_content (identical chunks fuse into one entry).
    scores: dict[str, tuple[float, Document]] = {}
    for ranked in ranked_lists:
        for rank_idx, doc in enumerate(ranked):
            key = doc.page_content
            prev_score, _ = scores.get(key, (0.0, doc))
            scores[key] = (prev_score + 1.0 / (k + rank_idx + 1), doc)

    ordered = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return ordered[:top_n]


def rerank(
    query: str, docs: list[Document], top_n: int = TOP_K_FINAL
) -> list[tuple[Document, float]]:
    """Rerank the docs via the reranker service and return the top_n."""
    if not docs:
        return []
    documents = [d.page_content for d in docs]
    resp = httpx.post(
        f"{RERANKER_URL}/rerank",
        json={
            "query": query,
            "documents": documents,
            "model": RERANKER_MODEL,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    # Infinity response: {"results": [{"relevance_score": float, "index": int, ...}, ...]}
    results = resp.json()["results"]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top = results[:top_n]
    return [(docs[item["index"]], item["relevance_score"]) for item in top]
