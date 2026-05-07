import logging
from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
COLLECTION_NAME = "langchain_rag"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


# Cache rationale: `get_vectorstore()` was instantiated on every retrieval and
# every ingestion batch, rebuilding the HTTP client and the OpenAIEmbeddings
# wrapper from scratch each time. Memoizing turns them into process-level
# singletons without changing the call sites.
@lru_cache(maxsize=1)
def _get_chroma_client() -> ClientAPI:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@lru_cache(maxsize=1)
def _get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Singleton Langchain-Chroma vectorstore bound to the service collection."""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def get_collection_count() -> int:
    return _get_chroma_client().get_collection(COLLECTION_NAME).count()


def reset_collection() -> None:
    """Drop the collection and invalidate the cached vectorstore.

    Used when the ingestion schema changes (e.g. deterministic chunk IDs) and
    existing embeddings need to be regenerated from scratch.
    """
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted collection '%s'", COLLECTION_NAME)
    except Exception:
        logger.info("Collection '%s' did not exist, skipping delete", COLLECTION_NAME)
    # Next call to get_vectorstore() will build a fresh Chroma wrapper against
    # the now-empty collection.
    get_vectorstore.cache_clear()
