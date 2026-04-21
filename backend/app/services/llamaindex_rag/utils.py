"""LlamaIndex-specific helpers.

Everything here is either coupled to a LlamaIndex type (`TextNode`,
`NodeWithScore`, `Document`) or only consumed by `LlamaindexSrv`. Kept out
of `app.services.utils` so the shared utils stay framework-agnostic.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import Document, NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI as LIOpenAI

from app.config import EXPANSION_MODEL, N_EXPANDED, TOP_K_AFTER_FUSION
from app.services.utils import rrf_fuse

# Anti-prompt-injection (sanitization layer 3 of the user question).
NONCE_BYTES = 8
MAX_QUESTION_CHARS = 2000


def expand_queries(msg: str) -> list[str]:
    """Generate N paraphrases of the query via LlamaIndex's OpenAI LLM."""
    llm = LIOpenAI(
        model=EXPANSION_MODEL,
        temperature=0.2,
        additional_kwargs={"response_format": {"type": "json_object"}},
    )
    system = (
        f"Eres un asistente que reescribe consultas para mejorar la recuperación. "
        f"Genera exactamente {N_EXPANDED} consultas alternativas (paráfrasis o ampliaciones "
        f"con sinónimos o enfoques distintos) que preserven el idioma de la consulta del usuario. "
        f'Devuelve ESTRICTAMENTE un JSON con la forma: {{"queries": ["q1", "q2"]}}. Sin texto extra.'
    )
    response = llm.chat(
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=msg),
        ]
    )
    data = json.loads(str(response.message.content))
    queries = data.get("queries", [])
    if len(queries) < N_EXPANDED:
        raise ValueError(f"Expansion returned {len(queries)} queries, expected {N_EXPANDED}")
    return queries[:N_EXPANDED]


def rrf_fuse_nodes(
    ranked_lists: list[list[NodeWithScore]], top_n: int = TOP_K_AFTER_FUSION
) -> list[tuple[float, NodeWithScore]]:
    """RRF over `NodeWithScore` lists; deduplicates by the node's text content."""
    return rrf_fuse(ranked_lists, key_fn=lambda n: n.node.get_content(), top_n=top_n)


def sanitize_question(question: str) -> str:
    """Strip control characters, collapse whitespace, cap length."""
    cleaned = "".join(
        ch for ch in question
        if unicodedata.category(ch)[0] != "C" or ch in "\t\n"
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned[:MAX_QUESTION_CHARS]


def format_nodes(nodes: list[NodeWithScore]) -> str:
    """Render the reranked nodes as the `CONTEXTO:` block of the prompt."""
    if not nodes:
        return "(sin contexto)"
    blocks = []
    for i, n in enumerate(nodes, 1):
        src = n.node.metadata.get("source", "?")
        page = n.node.metadata.get("page", "?")
        blocks.append(f"[{i}] source={src} page={page}\n{n.node.get_content()}")
    return "\n---\n".join(blocks)


def build_chunk_ids(chunks: list[TextNode]) -> list[str]:
    """Deterministic IDs: `{source}:{page}:{chunk_idx_within_page}`.

    Same scheme as the LangChain side — re-ingesting the same PDF upserts
    instead of duplicating chunks, and page-scoped indexing survives the
    insertion of a new page mid-book without invalidating the rest.
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


def pdf_page_to_document(text: str, source: str, page: int) -> Document:
    """Wrap a raw PDF page as a LlamaIndex `Document` with the right metadata."""
    return Document(text=text, metadata={"source": source, "page": page})
