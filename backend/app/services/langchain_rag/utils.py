"""LangChain-specific helpers.

Everything here either depends on a LangChain type (Document, ChatOpenAI) or
is only consumed by `LangchainSrv`. Kept out of `app.services.utils` so the
shared utils stay framework-agnostic.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import EXPANSION_MODEL, N_EXPANDED, TOP_K_AFTER_FUSION, TOP_K_FINAL
from app.services.utils import rerank_texts, rrf_fuse

# Anti-prompt-injection (layer 3: sanitize the user question).
NONCE_BYTES = 8
MAX_QUESTION_CHARS = 2000


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


def rrf_fuse_docs(
    ranked_lists: list[list[Document]], top_n: int = TOP_K_AFTER_FUSION
) -> list[tuple[float, Document]]:
    """RRF over LangChain Documents; deduplicates by `page_content`."""
    return rrf_fuse(ranked_lists, key_fn=lambda d: d.page_content, top_n=top_n)


def rerank(
    query: str, docs: list[Document], top_n: int = TOP_K_FINAL
) -> list[tuple[Document, float]]:
    """Rerank LangChain Documents via the shared Infinity helper."""
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    ranked = rerank_texts(query, texts, top_n=top_n)
    return [(docs[i], score) for i, score in ranked]


def sanitize_question(question: str) -> str:
    """Strip control characters, collapse whitespace, cap length."""
    cleaned = "".join(
        ch for ch in question
        if unicodedata.category(ch)[0] != "C" or ch in "\t\n"
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned[:MAX_QUESTION_CHARS]


def format_docs(docs: list[Document]) -> str:
    """Render the reranked chunks as the `CONTEXTO:` block of the prompt."""
    if not docs:
        return "(sin contexto)"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{i}] source={src} page={page}\n{d.page_content}")
    return "\n---\n".join(blocks)


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Deterministic IDs: `{source}:{page}:{chunk_idx_within_page}`.

    Re-ingesting the same PDF upserts instead of duplicating. Indexing per page
    (rather than globally) means adding a page to a PDF doesn't invalidate the
    IDs of chunks from unchanged pages.
    """
    per_page: Counter = Counter()
    ids: list[str] = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        page = c.metadata.get("page", 0)
        key = (source, page)
        idx = per_page[key]
        per_page[key] += 1
        ids.append(f"{source}:{page}:{idx}")
    return ids
