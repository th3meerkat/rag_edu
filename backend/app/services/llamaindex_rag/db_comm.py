"""LlamaIndex ↔ Chroma wiring.

All heavy objects are cached as process-level singletons: the HTTP Chroma
client, the `ChromaVectorStore` wrapper, the embedder, the LLM, and the
`VectorStoreIndex` bound to our collection. `get_index()` is the main entry
point consumed by `LlamaindexSrv`.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import EXPANSION_MODEL

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
COLLECTION_NAME = "llamaindex_rag"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


# ---------- Chroma client / collection ----------

@lru_cache(maxsize=1)
def _get_chroma_client() -> ClientAPI:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@lru_cache(maxsize=1)
def _get_chroma_collection():
    """Return the native Chroma collection (create if missing)."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )


# ---------- LlamaIndex global settings (shared models) ----------

@lru_cache(maxsize=1)
def _get_embed_model() -> BaseEmbedding:
    return OpenAIEmbedding(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_llm() -> LLM:
    return LIOpenAI(model=EXPANSION_MODEL, temperature=0.2, max_retries=3)


def _configure_settings() -> None:
    """Wire the global `Settings` once. LlamaIndex primitives read from here
    when no explicit `llm` / `embed_model` is passed."""
    if Settings.embed_model is None or not isinstance(Settings.embed_model, OpenAIEmbedding):
        Settings.embed_model = _get_embed_model()
    # Settings.llm is a lazy proxy by default; assign unconditionally (cheap).
    Settings.llm = _get_llm()


# ---------- Vector store / storage context / index ----------

@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Singleton `ChromaVectorStore` wrapping the native collection."""
    return ChromaVectorStore(chroma_collection=_get_chroma_collection())


@lru_cache(maxsize=1)
def get_storage_context() -> StorageContext:
    return StorageContext.from_defaults(vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_index() -> VectorStoreIndex:
    """Singleton `VectorStoreIndex` bound to the `llamaindex_rag` collection.

    Built from the existing vector store so it does not re-embed anything:
    the index is just a thin view over what lives in Chroma.
    """
    _configure_settings()
    return VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        storage_context=get_storage_context(),
        embed_model=_get_embed_model(),
    )


def get_collection_count() -> int:
    return _get_chroma_collection().count()


def reset_collection() -> None:
    """Drop the collection and invalidate every cached LlamaIndex wrapper.

    Used by the ingestion CLI's `--reset` flag when the chunking/ID schema
    changes and existing embeddings need to be regenerated from scratch.
    """
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted collection '%s'", COLLECTION_NAME)
    except Exception:
        logger.info("Collection '%s' did not exist, skipping delete", COLLECTION_NAME)
    # Wipe singletons so the next call rebuilds against the fresh collection.
    _get_chroma_collection.cache_clear()
    get_vector_store.cache_clear()
    get_storage_context.cache_clear()
    get_index.cache_clear()
