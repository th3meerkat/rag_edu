from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from app.config import DATA_DIR, ENV_PATH, TOP_K_FINAL
from app.services.utils import (
    build_filter,
    detect_positional,
    load_manifest,
    rerank,
    save_manifest,
)


class RagService(ABC):
    """Template Method for a RAG pipeline.

    Subclasses implement the framework-specific hooks:
      - _retrieve: vector search against the concrete vector store
      - _generate: LLM call with the retrieved context
      - _ingest:   PDF loading/chunking/embedding into the vector store
    """

    # ---------- Template methods (public) ----------

    def query(self, msg: str) -> str:
        """Full pipeline: detect → retrieve → (rrf) → rerank → generate."""
        load_dotenv(ENV_PATH)
        RagServiceLogs.log_query_start(msg)

        # --- Query expansion ---
        queries = self._expand_queries(msg)

        intent = detect_positional(msg)
        where: dict | None = None
        where_doc: dict | None = None
        if intent is not None:
            manifest = load_manifest()
            where, where_doc = build_filter(intent, manifest)
            RagServiceLogs.log_positional_detected(intent, manifest, where, where_doc)
        else:
            RagServiceLogs.log_positional_none()

        # --- Retrieve ---
        ranked_lists = [
            self._retrieve(q, where=where, where_document=where_doc) for q in queries
        ]

        # --- Reciprocal Rank Fusion (RRF) ---
        candidates = self._rrf(ranked_lists)

        # --- ReRank ---
        reranked = self._rerank(msg, candidates, top_n=TOP_K_FINAL)

        # --- Generate ---
        answer = self._generate(msg, [doc for doc, _ in reranked])
        RagServiceLogs.log_answer(answer)
        return answer

    def run_ingestion(self) -> None:
        """Ingest new PDFs according to the manifest; delegates the heavy lifting to _ingest."""
        load_dotenv(ENV_PATH)

        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            RagServiceLogs.log_no_pdfs()
            return

        manifest = load_manifest()
        new_pdf_paths = [p for p in pdf_paths if p.name not in manifest]
        if not new_pdf_paths:
            RagServiceLogs.log_no_new_pdfs(len(pdf_paths))
            return

        RagServiceLogs.log_new_pdfs_found(len(pdf_paths), len(new_pdf_paths))

        new_num_pages = self._ingest(new_pdf_paths)
        manifest.update(new_num_pages)
        save_manifest(manifest)
        RagServiceLogs.log_ingestion_complete()

    # ---------- Hooks (abstracts) ----------

    @abstractmethod
    def _ingest(self, pdf_paths: list[Path]) -> dict[str, int]:
        """Load/chunk/embed the PDFs into the vector store; returns {source: num_pages}."""
        ...

    @abstractmethod
    def _retrieve(
        self,
        query: str,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[Document]:
        """Filtered vector search; returns relevant chunks ordered by score."""
        ...

    @abstractmethod
    def _generate(self, question: str, docs: list[Document]) -> str:
        """Generate the final answer from the query and the already-reranked chunks."""
        ...


    # ---------- Utils ----------
    def _expand_queries(self, msg: str):
        # --- query expansion (desactivado temporalmente para simplificar pruebas/costo) ---
        # from app.utils import _expand_queries
        # expanded = _expand_queries(msg)
        # queries = [msg, *expanded]
        # print(f"[expand] {len(queries)} queries (1 original + {len(expanded)} expanded)")
        return [msg]

    def _rrf(self, ranked_lists):
        # --- RRF fusion (desactivado temporalmente; con una sola query no aporta) ---
        # from app.utils import _rrf_fuse, TOP_K_AFTER_FUSION, RRF_K
        # return _rrf_fuse(ranked_lists, k=RRF_K, top_n=TOP_K_AFTER_FUSION)
        # candidates = [doc for _, doc in fused]
        return ranked_lists[0]

    def _rerank(self, query: str, docs: list[Document], top_n: int = TOP_K_FINAL) -> list[tuple[Document, float]]:
        reranked = rerank(query, docs, top_n)
        RagServiceLogs.log_reranked(reranked)
        return reranked


class RagServiceLogs:
    """Logging helpers for RagService."""

    @staticmethod
    def log_query_start(msg: str) -> None:
        print(f"\n[query] user msg: {msg!r}")

    @staticmethod
    def log_positional_detected(intent, manifest, where, where_doc) -> None:
        print(
            f"[positional] intent={intent} manifest_sources={list(manifest)} "
            f"where={where} where_document={where_doc}"
        )

    @staticmethod
    def log_positional_none() -> None:
        print("[positional] no positional pattern detected")

    @staticmethod
    def log_answer(answer: str) -> None:
        print(f"[generate] {answer}")

    @staticmethod
    def log_no_pdfs() -> None:
        print(f"No PDFs found in {DATA_DIR}")

    @staticmethod
    def log_no_new_pdfs(total: int) -> None:
        print(f"No new PDFs to ingest ({total} already processed)")

    @staticmethod
    def log_new_pdfs_found(total: int, new: int) -> None:
        print(f"Found {total} PDF(s) in {DATA_DIR}; {new} new to ingest")

    @staticmethod
    def log_ingestion_complete() -> None:
        print("Ingestion complete.")

    @staticmethod
    def log_reranked(reranked: list[tuple[Document, float]]) -> None:
        print(f"[rerank] top-{TOP_K_FINAL}:")
        for r, (doc, score) in enumerate(reranked, 1):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            print(f"  {r}. rerank_score={score:.4f} source={src} page={page}")
