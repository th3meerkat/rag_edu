"""Framework-agnostic base class for the RAG services.

Keeps the *template method* hierarchy but stays completely independent from
LangChain and LlamaIndex: the chunk type each engine produces is a TypeVar `T`,
and the only I/O the base class does is globbing PDFs and reading/writing the
per-engine ingestion manifest.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from app.config import RAW_DATA_DIR
from app.services.utils import load_manifest, save_manifest

logger = logging.getLogger(__name__)

# Framework-native chunk / document / node type (e.g. LangChain `Document`,
# LlamaIndex `NodeWithScore`). The base class never inspects `T`.
T = TypeVar("T")


class RagService(ABC, Generic[T]):
    """Template Method for a RAG pipeline.

    Concrete subclasses pick a framework (LangChain, LlamaIndex, …) and plug
    it into three hooks:
      - `_ingest`:   PDF loading/chunking/embedding into the vector store.
      - `_retrieve`: filter-aware vector search; returns framework-native chunks.
      - `_generate`: LLM call that turns the reranked chunks into the answer.

    `run_ingestion` is a concrete template method reused by every engine.
    `query` is *intentionally abstract* — see its docstring below.
    """

    #: Short engine identifier. Drives the manifest filename and is echoed back
    #: to the client (e.g. for per-message badges in the UI).
    engine_name: ClassVar[str]

    # ---------- Template method (concrete) ----------

    def run_ingestion(self) -> None:
        """Ingest new PDFs according to the engine's manifest.

        Generic across engines: discover PDFs on disk, diff against the
        per-engine manifest, delegate the heavy lifting to `_ingest`, then
        merge the new page counts back into the manifest.
        """
        pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            logger.info("No PDFs found in %s", RAW_DATA_DIR)
            return

        manifest = load_manifest(self.engine_name)
        new_pdf_paths = [p for p in pdf_paths if p.name not in manifest]
        if not new_pdf_paths:
            logger.info("No new PDFs to ingest (%d already processed)", len(pdf_paths))
            return

        logger.info(
            "Found %d PDF(s) in %s; %d new to ingest",
            len(pdf_paths), RAW_DATA_DIR, len(new_pdf_paths),
        )

        new_num_pages = self._ingest(new_pdf_paths)
        manifest.update(new_num_pages)
        save_manifest(self.engine_name, manifest)
        logger.info("Ingestion complete.")

    # ---------- Template method (abstract — strategy shared, composition is not) ----------

    @abstractmethod
    def query(self, msg: str) -> str:
        """Run the end-to-end RAG pipeline and return the final answer.

        Every engine implements this in its own framework-idiomatic way, but
        they all follow the **same high-level five-step strategy**:

          1. **Expand** — paraphrase the user query into N variants. A single
             phrasing can miss relevant chunks; asking the same thing several
             ways widens recall for free.
          2. **Retrieve** — per query, a filter-aware vector search. A
             lightweight rules-based parser turns utterances like
             "página 5", "capítulo 3", "al final", "al principio" into
             Chroma `where` / `where_document` constraints, so positional
             questions target the right region of the PDFs instead of
             relying on semantic luck. See `utils.detect_positional` /
             `utils.build_filter`.
          3. **Fuse** — Reciprocal Rank Fusion (RRF) over the N ranked lists
             so chunks that surface near the top for *several* expansions
             bubble up and one-off noise falls.
          4. **Rerank** — a cross-encoder rerank (Infinity service) on the
             fused candidates. Trades a slow pass over a small set for a
             sharply better ordering than cosine alone.
          5. **Generate** — call the chat LLM with:
               (a) a system prompt carrying a layered anti-prompt-injection
                   defense: the user question travels isolated in its own
                   message, wrapped with a per-request random NONCE the
                   attacker cannot close; sanitization strips control chars
                   and caps length.
               (b) a short-term conversational memory window (last ~5 turns)
                   so anaphoric follow-ups ("¿y al final qué pasa?") make
                   sense.

        **The strategy is shared; the composition primitives are not.**
        Each subclass is free to realise the pipeline with whatever its
        framework makes idiomatic:

          - `LangchainSrv.query` composes the five steps as a single LCEL
            `Runnable` (`prepare → retrieve/fuse → rerank → generate`),
            unlocking `.stream` / `.ainvoke` / `.batch`, retries, fallbacks
            and per-step LangSmith spans for free.
          - `LlamaindexSrv.query` composes them as a native `CustomQueryEngine`
            with a `BaseNodePostprocessor` for the rerank step and a
            `ChatMemoryBuffer` for the memory window.

        Both implementations drive the same abstract hooks (`_retrieve`,
        `_generate`) so the extension points survive polymorphism.
        """

    # ---------- Hooks (abstract) ----------

    @abstractmethod
    def _ingest(self, pdf_paths: list[Path]) -> dict[str, int]:
        """Load/chunk/embed the PDFs into the vector store; returns {source: num_pages}."""

    @abstractmethod
    def _retrieve(
        self,
        query: str,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[T]:
        """Filtered vector search; returns relevant chunks ordered by score."""

    @abstractmethod
    def _generate(self, question: str, docs: list[T]) -> str:
        """Generate the final answer from the query and the reranked chunks."""
