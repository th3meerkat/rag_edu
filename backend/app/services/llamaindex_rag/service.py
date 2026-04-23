"""LlamaIndex implementation of the RAG service.

Leans on as much LlamaIndex machinery as possible:

  * `PDFReader` + `SentenceSplitter` for ingestion parsing/chunking.
  * `VectorStoreIndex` bound to a `ChromaVectorStore` for embedding + storage.
  * A `BaseRetriever` subclass that queries the native Chroma collection
    directly so Chroma's `where` / `where_document` filter dialect is
    available without having to translate to LlamaIndex `MetadataFilters`.
  * `InfinityRerank(BaseNodePostprocessor)` for the rerank step.
  * `ChatMemoryBuffer` for the short-term conversational memory.
  * `CustomQueryEngine` as the top-level composition primitive — the
    framework-native equivalent of LangChain's LCEL runnable.
"""
from __future__ import annotations

import logging
import secrets
import tiktoken
from functools import cached_property, lru_cache
from pathlib import Path
from typing import ClassVar, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.readers.file import PDFReader
from pydantic import ConfigDict

from app.config import EXPANSION_MODEL, TOP_K_FINAL, TOP_K_PER_QUERY
from app.services.llamaindex_rag.db_comm import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    _get_chroma_collection,
    _get_embed_model,
    _get_llm,
    get_collection_count,
    get_index,
)
from app.services.llamaindex_rag.memory import (
    append_turn,
    recent_messages,
)
from app.services.llamaindex_rag.postprocessor import InfinityRerank
from app.services.llamaindex_rag.prompts import build_prompt
from app.services.llamaindex_rag.utils import (
    NONCE_BYTES,
    build_chunk_ids,
    format_nodes,
    sanitize_question,
)
from app.services.rag import RagService
from app.services.utils import build_filter, detect_positional, load_manifest

# Same chunking targets as the LangChain side: 500/100 token chunks aligned
# with the embedding model's tokenizer.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

# Ingestion batch: `insert_nodes` ultimately calls the embeddings API; 256 is
# well below OpenAI's 2048-input limit and keeps a retry after a transient
# failure cheap. Matches the LangChain side for parity.
INGEST_BATCH_SIZE = 256

logger = logging.getLogger(__name__)


def _log_retrieved_nodes(hits: list[NodeWithScore]) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    for r, n in enumerate(hits, 1):
        src = n.node.metadata.get("source", "?")
        page = n.node.metadata.get("page", "?")
        snippet = n.node.get_content()[:80].replace("\n", " ")
        logger.info("    %d. source=%s page=%s | %s…", r, src, page, snippet)


@lru_cache(maxsize=1)
def _tokenizer_fn():
    """Tiktoken encoder aligned with the embedding model."""
    return tiktoken.encoding_for_model(EMBEDDING_MODEL).encode


# ---------- Custom retriever: native Chroma `where` / `where_document` support ----------

class ChromaFilterRetriever(BaseRetriever):
    """Retriever that talks to the raw Chroma collection.

    LlamaIndex's stock `VectorIndexRetriever` takes a `MetadataFilters`
    object, which is great for the usual `field == value` case but does not
    expose Chroma's `$contains` full-text filter (what we use for
    `capítulo N`). Going one level below and hitting the collection's
    `.query()` gives us the full Chroma dialect back — the same filters
    built by `app.services.utils.build_filter` that the LangChain side
    already consumes.
    """

    def __init__(
        self,
        similarity_top_k: int = TOP_K_PER_QUERY,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ):
        super().__init__()
        self._similarity_top_k = similarity_top_k
        self._where = where
        self._where_document = where_document

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Embed the query via the configured embedder so we stay consistent
        # with ingestion (same model, same normalization).
        embed_model = _get_embed_model()
        query_embedding = embed_model.get_query_embedding(query_bundle.query_str)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": self._similarity_top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if self._where is not None:
            kwargs["where"] = self._where
        if self._where_document is not None:
            kwargs["where_document"] = self._where_document

        result = _get_chroma_collection().query(**kwargs)

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        nodes: list[NodeWithScore] = []
        for i, text in enumerate(docs):
            node = TextNode(
                id_=ids[i] if i < len(ids) else None,
                text=text,
                metadata=metas[i] or {},
            )
            # Chroma returns a distance (cosine distance in our setup); convert
            # to a similarity-style score so downstream components treat
            # "higher is better" consistently with the LangChain path.
            score = 1.0 - float(dists[i]) if i < len(dists) else None
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes


# ---------- CustomQueryEngine: five-step strategy, LlamaIndex-native composition ----------

class RagQueryEngine(CustomQueryEngine):
    """LlamaIndex `CustomQueryEngine` composing the shared 5-step strategy.

    Why a `CustomQueryEngine` (vs. stitching calls by hand in the service):
    it makes the pipeline a first-class LlamaIndex component — gets
    callback/instrumentation for free, plugs into `QueryPipeline` if the
    caller wants, and uses the framework's `response_gen` conventions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pydantic fields — LlamaIndex query engines are pydantic models.
    llm: LLM
    reranker: InfinityRerank
    retriever_factory_where: Optional[dict] = None
    retriever_factory_where_document: Optional[dict] = None
    engine_name: str = "llamaindex"

    def custom_query(self, query_str: str) -> str:
        # Step 1: expand (disabled for now; see utils.expand_queries).
        queries = [query_str]

        # Step 2: retrieve (one retrieval per expanded query).
        ranked_lists: list[list[NodeWithScore]] = []
        for q in queries:
            retriever = ChromaFilterRetriever(
                similarity_top_k=TOP_K_PER_QUERY,
                where=self.retriever_factory_where,
                where_document=self.retriever_factory_where_document,
            )
            logger.info(
                "[retrieve] cosine, top-%d (where=%s where_document=%s)",
                TOP_K_PER_QUERY,
                self.retriever_factory_where,
                self.retriever_factory_where_document,
            )
            hits = retriever.retrieve(q)
            logger.info("  → %d hits", len(hits))
            _log_retrieved_nodes(hits)
            ranked_lists.append(hits)

        # Step 3: RRF fuse. Single-query → no-op (would just reorder by its
        # own rank). Kept as the explicit branch so the pipeline reads like
        # the docstring of `RagService.query`.
        if len(ranked_lists) == 1:
            candidates = ranked_lists[0]
        else:
            from app.services.llamaindex_rag.utils import rrf_fuse_nodes
            candidates = [node for _, node in rrf_fuse_nodes(ranked_lists)]

        # Step 4: rerank via the native postprocessor.
        reranked = self.reranker.postprocess_nodes(
            candidates, query_bundle=QueryBundle(query_str=query_str)
        )

        # Step 5: generate with anti-injection + short-term memory.
        safe_question = sanitize_question(query_str)
        nonce = secrets.token_hex(NONCE_BYTES)
        context = format_nodes(reranked)

        logger.info(
            "[generate] model=%s nonce=%s question_len=%d",
            EXPANSION_MODEL, nonce, len(safe_question),
        )

        messages = build_prompt(
            context=context,
            nonce=nonce,
            question=safe_question,
            history=recent_messages(),
        )
        response = self.llm.chat(messages=messages)
        answer = str(response.message.content or "")

        # Persist the turn *after* a successful generation so a mid-pipeline
        # error doesn't corrupt the conversation window.
        append_turn(safe_question, answer)
        return answer


# ---------- Service ----------

class LlamaindexSrv(RagService[NodeWithScore]):
    engine_name: ClassVar[str] = "llamaindex"

    # --- _ingest ---

    def _ingest(self, pdf_paths: list[Path]) -> dict[str, int]:
        # Step 1: parse each PDF into per-page Documents. PDFReader returns
        # `page_label` (1-indexed); normalize to `page` (0-indexed) + `source`
        # so the metadata matches what the LangChain side writes and what
        # `build_filter` expects.
        from llama_index.core.schema import Document as LIDocument

        documents: list[LIDocument] = []
        new_num_pages: dict[str, int] = {}
        reader = PDFReader()
        for pdf_path in pdf_paths:
            pages = reader.load_data(pdf_path)
            for page_doc in pages:
                page_label = page_doc.metadata.get("page_label", "1")
                try:
                    page_idx = int(page_label) - 1
                except ValueError:
                    page_idx = 0
                page_doc.metadata = {"source": pdf_path.name, "page": page_idx}
            documents.extend(pages)
            new_num_pages[pdf_path.name] = len(pages)
            logger.info("Loaded %s: %d page(s)", pdf_path.name, len(pages))

        # Step 2: token-aware chunking aligned with the embedding tokenizer.
        splitter = SentenceSplitter(
            chunk_size=CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            tokenizer=_tokenizer_fn(),
        )
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info("Generated %d chunk(s)", len(nodes))

        # Step 3: stamp deterministic IDs on every node so re-ingestion upserts.
        ids = build_chunk_ids([TextNode(text=n.get_content(), metadata=n.metadata) for n in nodes])
        for n, nid in zip(nodes, ids):
            n.id_ = nid

        # Step 4: batched insert into the vector store via the index.
        index = get_index()
        for start in range(0, len(nodes), INGEST_BATCH_SIZE):
            end = start + INGEST_BATCH_SIZE
            index.insert_nodes(nodes[start:end])

        logger.info(
            "Collection '%s' has %d embedding(s) (space=cosine).",
            COLLECTION_NAME, get_collection_count(),
        )
        return new_num_pages

    # --- _retrieve ---

    def _retrieve(
        self,
        query: str,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[NodeWithScore]:
        logger.info(
            "[retrieve] cosine, top-%d (where=%s where_document=%s)",
            TOP_K_PER_QUERY, where, where_document,
        )
        retriever = ChromaFilterRetriever(
            similarity_top_k=TOP_K_PER_QUERY,
            where=where,
            where_document=where_document,
        )
        hits = retriever.retrieve(query)

        logger.info("  → %d hits", len(hits))
        _log_retrieved_nodes(hits)
        return hits

    # --- _generate ---

    def _generate(self, question: str, docs: list[NodeWithScore]) -> str:
        """Generate an answer given pre-reranked nodes.

        This hook mirrors the LangChain counterpart: it receives the already
        reranked nodes and only assembles the prompt, calls the LLM, and
        updates the memory. `query()` is the idiomatic entry point; this
        hook exists to satisfy the abstract template and to be testable.
        """
        safe_question = sanitize_question(question)
        nonce = secrets.token_hex(NONCE_BYTES)
        context = format_nodes(docs)

        logger.info(
            "[generate] model=%s nonce=%s question_len=%d",
            EXPANSION_MODEL, nonce, len(safe_question),
        )

        messages = build_prompt(
            context=context,
            nonce=nonce,
            question=safe_question,
            history=recent_messages(),
        )
        response = _get_llm().chat(messages=messages)
        answer = str(response.message.content or "")
        append_turn(safe_question, answer)
        logger.info("[generate] %s", answer)
        return answer

    # --- query() override: native LlamaIndex composition ---

    def query(self, msg: str) -> str:
        """Full RAG pipeline as a `CustomQueryEngine`. See `RagService.query`."""
        logger.info("[query] user msg: %r", msg)

        # Prepare the Chroma filters up front so the QueryEngine is rebuilt
        # per-request (filters depend on the user message). Cheap — the heavy
        # state lives in module-level singletons (index, llm, embedder).
        intent = detect_positional(msg)
        where: dict | None = None
        where_doc: dict | None = None
        if intent is not None:
            manifest = load_manifest(self.engine_name)
            where, where_doc = build_filter(intent, manifest)
            logger.info(
                "[positional] intent=%s manifest_sources=%s where=%s where_document=%s",
                intent, list(manifest), where, where_doc,
            )
        else:
            logger.info("[positional] no positional pattern detected")

        engine = self._build_query_engine(where=where, where_document=where_doc)
        answer = engine.custom_query(msg)
        logger.info("[generate] %s", answer)
        return answer

    def _build_query_engine(
        self, where: dict | None, where_document: dict | None
    ) -> RagQueryEngine:
        return RagQueryEngine(
            llm=_get_llm(),
            reranker=InfinityRerank(top_n=TOP_K_FINAL),
            retriever_factory_where=where,
            retriever_factory_where_document=where_document,
            engine_name=self.engine_name,
        )

    @cached_property
    def _index(self) -> VectorStoreIndex:
        """The shared LlamaIndex vector-store index. Cached per instance so
        tests can patch `get_index` without the constructor capturing a
        stale reference."""
        return get_index()
