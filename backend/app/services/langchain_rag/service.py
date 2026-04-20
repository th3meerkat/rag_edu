import logging
import re
import secrets
import unicodedata
from collections import Counter
from functools import cached_property, lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import TOP_K_FINAL, TOP_K_PER_QUERY
from app.services.langchain_rag.db_comm import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_collection_count,
    get_vectorstore,
)
from app.services.langchain_rag.prompts import build_prompt
from app.services.rag import RagService
from app.services.utils import (
    EXPANSION_MODEL,
    build_filter,
    detect_positional,
    load_manifest,
)

# Chunking in tokens (not characters) aligns chunks with the embedding model's
# tokenizer: `text-embedding-3-small` accepts 8191 tokens per input, so 500
# leaves plenty of headroom and keeps chunks semantically coherent. Character
# counting under/over-chunks by up to 4x depending on language.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

# Ingestion runs in batches to cap each OpenAI embeddings request. A large book
# can produce thousands of chunks; sending them all in a single `add_documents`
# call risks request-size limits and timeouts. The size (256) is a safe default
# for `text-embedding-3-small`: well below the 2048-input API limit and small
# enough that a retry after a transient failure is cheap.
INGEST_BATCH_SIZE = 256

# Prompt-injection defense without a second LLM:
#  - Layer 1: the question travels in its OWN HumanMessage (role separation).
#  - Layer 2: the question is wrapped with <question_{nonce}>...</question_{nonce}>;
#    the nonce is fresh per request so an attacker can't close the block.
#  - Layer 3: sanitize input (strip control chars, collapse whitespace, cap len).
NONCE_BYTES = 8
MAX_QUESTION_CHARS = 2000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    """Process-level LLM client. `max_retries` covers transient 5xx/timeouts,
    which are the most common failure mode of the OpenAI API."""
    return ChatOpenAI(model=EXPANSION_MODEL, temperature=0.2, max_retries=3)


@lru_cache(maxsize=1)
def _get_chain() -> Runnable:
    """LCEL chain: prompt | llm | parser.

    Using LCEL instead of manual `llm.invoke([messages])` gives us streaming,
    ainvoke, batch, LangSmith tracing, with_retry and with_fallbacks for free.
    `StrOutputParser` also handles AIMessage.content when it's a list of
    content blocks (multimodal, tool-calls) — `str(response.content)` does not.
    """
    return build_prompt() | _get_llm() | StrOutputParser()


def _sanitize_question(question: str) -> str:
    """Strip control characters, collapse whitespace, cap length."""
    cleaned = "".join(
        ch for ch in question
        if unicodedata.category(ch)[0] != "C" or ch in "\t\n"
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned[:MAX_QUESTION_CHARS]


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(sin contexto)"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{i}] source={src} page={page}\n{d.page_content}")
    return "\n---\n".join(blocks)


def _build_chunk_ids(chunks: list[Document]) -> list[str]:
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


class LangchainSrv(RagService):
    def _ingest(self, pdf_paths: list[Path]) -> dict[str, int]:
        # --- Open and parse files ---
        documents: list[Document] = []
        new_num_pages: dict[str, int] = {}
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_path.name
            documents.extend(pages)
            new_num_pages[pdf_path.name] = len(pages)
            logger.info("Loaded %s: %d page(s)", pdf_path.name, len(pages))

        # --- Chunk by tokens (tiktoken-backed) ---
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=EMBEDDING_MODEL,
            chunk_size=CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )
        chunks = splitter.split_documents(documents)
        logger.info("Generated %d chunk(s)", len(chunks))

        # --- Embed + store, idempotent and batched ---
        # Deterministic IDs (upsert on reingest) and batching (cap each OpenAI
        # embeddings request) are tightly coupled: we already need to iterate
        # to assign IDs, so slicing into batches is free in the same loop.
        ids = _build_chunk_ids(chunks)
        vectorstore = get_vectorstore()
        for start in range(0, len(chunks), INGEST_BATCH_SIZE):
            end = start + INGEST_BATCH_SIZE
            vectorstore.add_documents(chunks[start:end], ids=ids[start:end])

        logger.info(
            "Collection '%s' has %d embedding(s) (space=cosine).",
            COLLECTION_NAME, get_collection_count(),
        )
        return new_num_pages

    def _retrieve(
        self,
        query: str,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[Document]:
        logger.info(
            "[retrieve] cosine, top-%d (where=%s where_document=%s)",
            TOP_K_PER_QUERY, where, where_document,
        )

        # Idiomatic Langchain: build a Retriever, not a raw similarity call.
        # Retrievers are Runnables (composable with LCEL) and accept both the
        # metadata `filter` and Chroma's `where_document` via search_kwargs.
        search_kwargs: dict = {"k": TOP_K_PER_QUERY}
        if where is not None:
            search_kwargs["filter"] = where
        if where_document is not None:
            search_kwargs["where_document"] = where_document

        retriever = get_vectorstore().as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)

        logger.info("  → %d hits", len(docs))
        for r, d in enumerate(docs, 1):
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "?")
            snippet = d.page_content[:80].replace("\n", " ")
            logger.info("    %d. source=%s page=%s | %s…", r, src, page, snippet)
        return docs

    def _generate(self, question: str, docs: list[Document]) -> str:
        # Three layers of prompt-injection defense, no extra LLM:
        # 1) Structural separation — the question goes in its own HumanMessage.
        # 2) Random-nonce delimiter — <question_NONCE>...</question_NONCE> with
        #    a per-request nonce, so an attacker can't close the block.
        # 3) Sanitization — strip control chars, collapse whitespace, cap length.
        safe_question = _sanitize_question(question)
        nonce = secrets.token_hex(NONCE_BYTES)
        context = _format_docs(docs)

        logger.info(
            "[generate] model=%s nonce=%s question_len=%d",
            EXPANSION_MODEL, nonce, len(safe_question),
        )

        return _get_chain().invoke(
            {"context": context, "nonce": nonce, "question": safe_question}
        )

    # ---------- Langchain-native override of the parent's template method ----------
    #
    # Rationale: the parent implements `query` imperatively (good default for a
    # framework-neutral base). Here we compose the exact same pipeline as a
    # single LCEL Runnable so the pipeline *itself* is a first-class Langchain
    # object — callers gain `.stream(msg)`, `.ainvoke(msg)`, `.batch([msgs])`,
    # `.with_retry()`, `.with_fallbacks()`, and per-step LangSmith spans for
    # free, without any extra code. Subclass hooks (`_retrieve`, `_generate`,
    # plus the parent's `_expand_queries`, `_rrf`, `_rerank`) are still invoked
    # inside each step, so the extension points survive polymorphism.

    def query(self, msg: str) -> str:
        """Full RAG pipeline as a LCEL Runnable. Polymorphic override."""
        logger.info("[query] user msg: %r", msg)
        return self._query_chain.invoke(msg)

    @cached_property
    def _query_chain(self) -> Runnable:
        """Build the chain once per instance and reuse.

        Shape: prepare_state → retrieve_and_fuse → rerank → generate.
        State threads through as a dict; `RunnablePassthrough.assign` adds keys
        while preserving prior ones, so each step sees everything upstream
        produced. `.with_config(run_name=...)` gives each span a meaningful
        name in LangSmith traces instead of a generic "RunnableLambda".
        """
        return (
            RunnableLambda(self._prepare_state)
                .with_config(run_name="prepare_state")
            | RunnablePassthrough.assign(
                candidates=RunnableLambda(self._retrieve_and_fuse)
                    .with_config(run_name="retrieve_and_fuse")
            )
            | RunnablePassthrough.assign(
                docs=RunnableLambda(self._rerank_candidates)
                    .with_config(run_name="rerank")
            )
            | RunnableLambda(self._generate_answer)
                .with_config(run_name="generate")
        )

    # --- LCEL step implementations ---
    # Each step receives the accumulated state dict and returns either a slice
    # of new state (consumed by `RunnablePassthrough.assign`) or the final
    # answer (terminal step).

    def _prepare_state(self, msg: str) -> dict:
        """Entry: build the state dict and detect positional filters."""
        intent = detect_positional(msg)
        if intent is None:
            logger.info("[positional] no positional pattern detected")
            return {"msg": msg, "where": None, "where_document": None}
        manifest = load_manifest()
        where, where_doc = build_filter(intent, manifest)
        logger.info(
            "[positional] intent=%s manifest_sources=%s where=%s where_document=%s",
            intent, list(manifest), where, where_doc,
        )
        return {"msg": msg, "where": where, "where_document": where_doc}

    def _retrieve_and_fuse(self, state: dict) -> list[Document]:
        """Expand the query, fan out retrieval via `.batch()`, then RRF-fuse.

        `.batch()` is LCEL's native parallel dispatch — once `_expand_queries`
        is re-enabled, the N retrievals run concurrently (threadpool for sync
        / asyncio for async) without us writing a single await or future.
        """
        queries = self._expand_queries(state["msg"])
        where, where_doc = state["where"], state["where_document"]

        def retrieve_one(q: str) -> list[Document]:
            return self._retrieve(q, where=where, where_document=where_doc)

        retriever: Runnable[str, list[Document]] = RunnableLambda(retrieve_one)
        ranked_lists = retriever.batch(queries)
        
        # --- Reciprocal Rank Fusion (RRF) ---
        return self._rrf(ranked_lists)

    def _rerank_candidates(self, state: dict) -> list[Document]:
        reranked = self._rerank(state["msg"], state["candidates"], top_n=TOP_K_FINAL)
        return [doc for doc, _ in reranked]

    def _generate_answer(self, state: dict) -> str:
        answer = self._generate(state["msg"], state["docs"])
        logger.info("[generate] %s", answer)
        return answer
