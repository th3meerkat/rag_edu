"""Shared fixtures for regression tests.

All external integrations are mocked except for real ChromaDB (test collection)
and OpenAI embeddings (cheap, deterministic enough for retrieval tests).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import pytest
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import OpenAIEmbeddings

from app.config import ENV_PATH
from app.services.langchain_rag.db_comm import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_METADATA,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


TEST_COLLECTION_NAME = "langchain_rag_test"


@pytest.fixture(scope="session", autouse=True)
def _load_env():
    """Ensure OPENAI_API_KEY (and friends) are available for tests that use
    real embeddings against ChromaDB.
    """
    load_dotenv(ENV_PATH)


# ---------- Data fixtures ----------

@pytest.fixture
def fake_docs() -> list[Document]:
    """A small deterministic list of Document objects."""
    return [
        Document(
            page_content="El principito habla con el zorro en el capítulo 21.",
            metadata={"source": "el_principito.pdf", "page": 20},
        ),
        Document(
            page_content="Capítulo 1: Cuando tenía seis años vi un dibujo.",
            metadata={"source": "el_principito.pdf", "page": 0},
        ),
        Document(
            page_content="Al final, el principito regresa a su planeta.",
            metadata={"source": "el_principito.pdf", "page": 46},
        ),
    ]


@pytest.fixture
def fake_manifest() -> dict[str, int]:
    return {"el_principito.pdf": 47, "otro_libro.pdf": 100}


# ---------- Filesystem fixtures ----------

@pytest.fixture
def temp_manifest_path(monkeypatch, tmp_path) -> Path:
    """Redirect INGESTED_MANIFEST to a tmp_path for isolated I/O tests."""
    fake_path = tmp_path / "ingested.json"
    monkeypatch.setattr("app.services.utils.INGESTED_MANIFEST", fake_path)
    return fake_path


# ---------- LLM / HTTP mocks ----------

class _EchoLLM(Runnable):
    """Fake Runnable chat model: records invocations and returns a fixed AIMessage.

    Must be a Runnable so it composes inside LCEL chains (`prompt | llm | parser`).
    """

    def __init__(self, content: str):
        super().__init__()
        self._content = content
        self.invoke_calls: list[list[Any]] = []

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        if hasattr(input, "to_messages"):
            messages = input.to_messages()
        elif isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = list(input)
        self.invoke_calls.append(messages)
        return AIMessage(content=self._content)


@pytest.fixture
def echo_chat_openai(monkeypatch):
    """Patch ChatOpenAI constructors used by the code. Returns a factory
    that captures the last instance per module for assertions.
    """
    created: dict[str, _EchoLLM] = {}

    def make_factory(module_path: str, default_content: str):
        def factory(*args, **kwargs):
            llm = _EchoLLM(default_content)
            created[module_path] = llm
            return llm
        monkeypatch.setattr(f"{module_path}.ChatOpenAI", factory)

    # utils.expand_queries: must return JSON with 2 queries (N_EXPANDED=2)
    make_factory(
        "app.services.utils",
        json.dumps({"queries": ["q-alt-1", "q-alt-2"]}),
    )
    # langchain_rag.service._generate: any text is fine
    make_factory(
        "app.services.langchain_rag.service",
        "respuesta-eco",
    )

    # The service caches its LLM/chain in lru_cache; clear so the patch applies.
    # Also wipe the short-term memory singleton so history doesn't leak across tests.
    from app.services.langchain_rag import service as _svc
    from app.services.langchain_rag.memory import clear_history
    _svc._get_llm.cache_clear()
    _svc._get_chain.cache_clear()
    clear_history()

    yield created

    _svc._get_llm.cache_clear()
    _svc._get_chain.cache_clear()
    clear_history()


@pytest.fixture
def mock_reranker(monkeypatch):
    """Patch httpx.post used by utils.rerank. Returns scores = 1/(i+1) keeping order."""
    calls: list[dict] = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "json": json, "timeout": timeout})
        docs = json["documents"]
        results = [
            {"index": i, "relevance_score": 1.0 / (i + 1)}
            for i in range(len(docs))
        ]
        return _Resp({"results": results})

    monkeypatch.setattr("app.services.utils.httpx.post", fake_post)
    return calls


# ---------- PDF loader mock ----------

@pytest.fixture
def mock_pypdf_loader(monkeypatch):
    """Patch PyPDFLoader so it returns canned pages indexed by filename."""
    pages_by_name: dict[str, list[Document]] = {}

    class _FakeLoader:
        def __init__(self, path: str):
            self._path = Path(path)

        def load(self):
            name = self._path.name
            return pages_by_name.get(name, [
                Document(page_content=f"page 0 of {name}", metadata={"page": 0}),
                Document(page_content=f"page 1 of {name}", metadata={"page": 1}),
            ])

    monkeypatch.setattr(
        "app.services.langchain_rag.service.PyPDFLoader", _FakeLoader
    )
    return pages_by_name


# ---------- Real ChromaDB test collection ----------

@pytest.fixture(scope="session")
def chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@pytest.fixture(scope="session")
def test_collection_seeded(chroma_client):
    """Create langchain_rag_test as a copy of langchain_rag (session-scoped).

    Destroyed at the end of the session so langchain_rag stays intact.
    """
    # Clean any stale test collection from a previous run.
    try:
        chroma_client.delete_collection(TEST_COLLECTION_NAME)
    except Exception:
        pass

    test_col = chroma_client.create_collection(
        name=TEST_COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

    # Copy from the production collection if it exists.
    try:
        src = chroma_client.get_collection(COLLECTION_NAME)
        data = src.get(include=["documents", "metadatas", "embeddings"])
        if data["ids"]:
            test_col.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"],
            )
    except Exception:
        # Source missing: leave the test collection empty.
        pass

    yield test_col

    try:
        chroma_client.delete_collection(TEST_COLLECTION_NAME)
    except Exception:
        pass


@pytest.fixture
def test_vectorstore(chroma_client, test_collection_seeded) -> Chroma:
    """A Chroma vectorstore (langchain) pointing at the test collection."""
    return Chroma(
        client=chroma_client,
        collection_name=TEST_COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata=COLLECTION_METADATA,
    )


@pytest.fixture
def patched_langchain_db(monkeypatch, chroma_client, test_vectorstore):
    """Redirect LangchainSrv's db_conn helpers to the test collection."""
    monkeypatch.setattr(
        "app.services.langchain_rag.service.get_vectorstore",
        lambda: test_vectorstore,
    )
    monkeypatch.setattr(
        "app.services.langchain_rag.service.get_collection_count",
        lambda: chroma_client.get_collection(TEST_COLLECTION_NAME).count(),
    )
    return test_vectorstore


@pytest.fixture
def empty_test_collection(chroma_client, test_collection_seeded):
    """Same collection but emptied, for ingest tests that need a blank slate.

    Restores the seeded state after the test by re-copying from source.
    """
    all_ids = test_collection_seeded.get()["ids"]
    if all_ids:
        test_collection_seeded.delete(ids=all_ids)
    yield test_collection_seeded
    # Re-seed from source.
    try:
        src = chroma_client.get_collection(COLLECTION_NAME)
        data = src.get(include=["documents", "metadatas", "embeddings"])
        if data["ids"]:
            current = test_collection_seeded.get()["ids"]
            if current:
                test_collection_seeded.delete(ids=current)
            test_collection_seeded.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"],
            )
    except Exception:
        pass
