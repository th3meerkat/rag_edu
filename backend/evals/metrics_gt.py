"""Ground-truth retrieval metrics — deterministic, no API calls.

All metrics operate on *retrieved pages* vs a set of *relevant pages*. Page is
the chosen granularity because LangChain / LlamaIndex chunk differently; the
book itself is the shared reference frame.
"""
from __future__ import annotations


def hit_at_k(retrieved_pages: list[int], relevant_pages: set[int]) -> float:
    """1.0 if any of the top-k retrieved pages is relevant, else 0.0."""
    if not retrieved_pages:
        return 0.0
    return 1.0 if any(p in relevant_pages for p in retrieved_pages) else 0.0


def recall_at_k(retrieved_pages: list[int], relevant_pages: set[int]) -> float:
    """Fraction of relevant pages covered by the top-k retrieved pages."""
    if not relevant_pages:
        return 0.0
    covered = {p for p in retrieved_pages if p in relevant_pages}
    return len(covered) / len(relevant_pages)


def reciprocal_rank(retrieved_pages: list[int], relevant_pages: set[int]) -> float:
    """1/rank of the first relevant page in the retrieved list, else 0.0."""
    for i, p in enumerate(retrieved_pages, start=1):
        if p in relevant_pages:
            return 1.0 / i
    return 0.0
