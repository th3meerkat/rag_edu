"""LlamaIndex-native rerank postprocessor backed by the Infinity service.

Slots into a `QueryEngine` via `node_postprocessors=[InfinityRerank(...)]`
and reorders / trims the retrieved `NodeWithScore` list. Using the shared
`rerank_texts` helper keeps the HTTP wire contract in one place; the adapter
here is only concerned with mapping indexes back to LlamaIndex types.
"""
from __future__ import annotations

import logging
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.config import TOP_K_FINAL
from app.services.utils import rerank_texts

logger = logging.getLogger(__name__)


class InfinityRerank(BaseNodePostprocessor):
    """Cross-encoder rerank via the Infinity service.

    Attributes:
      top_n: max number of nodes kept after reranking.
    """

    top_n: int = TOP_K_FINAL

    @classmethod
    def class_name(cls) -> str:
        return "InfinityRerank"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        texts = [n.node.get_content() for n in nodes]
        ranked = rerank_texts(query_bundle.query_str, texts, top_n=self.top_n)

        logger.info("[rerank] top-%d:", self.top_n)
        reranked: list[NodeWithScore] = []
        for r, (idx, score) in enumerate(ranked, 1):
            node = nodes[idx].node
            src = node.metadata.get("source", "?")
            page = node.metadata.get("page", "?")
            logger.info("  %d. rerank_score=%.4f source=%s page=%s", r, score, src, page)
            reranked.append(NodeWithScore(node=node, score=score))
        return reranked
