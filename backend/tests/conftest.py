"""Shared fixtures for regression tests.

All external integrations are mocked except for real ChromaDB (test collections)
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
from app.services.llamaindex_rag.db_comm import (
    COLLECTION_NAME as LI_COLLECTION_NAME,
)


TEST_COLLECTION_NAME = "langchain_rag_test"
LI_TEST_COLLECTION_NAME = "llamaindex_rag_test"


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
def manifest_dir(monkeypatch, tmp_path) -> Path:
    """Redirect the shared `DATA_DIR` used by the manifest helpers to tmp_path."""
    monkeypatch.setattr("app.services.utils.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.services.rag.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def temp_manifest_path(manifest_dir) -> Path:
    """Path the tests treat as "the manifest" — one per test, one engine-key.

    Tests that only exercise utils / rag template methods use engine "test".
    """
    return manifest_dir / "ingested_test.json"


# ---------- LLM / HTTP mocks (LangChain) ----------

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
    """Patch ChatOpenAI constructors used by the LangChain code path. Returns
    a dict that captures the last instance per module for assertions.
    """
    created: dict[str, _EchoLLM] = {}

    def make_factory(module_path: str, default_content: str):
        def factory(*args, **kwargs):
            llm = _EchoLLM(default_content)
            created[module_path] = llm
            return llm
        monkeypatch.setattr(f"{module_path}.ChatOpenAI", factory)

    # langchain_rag.utils.expand_queries: must return JSON with 2 queries (N_EXPANDED=2)
    make_factory(
        "app.services.langchain_rag.utils",
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


# ---------- LLM mocks (LlamaIndex) ----------

class _EchoLIResponse:
    """Shape returned by `LLM.chat`: has `.message.content`."""

    def __init__(self, content: str):
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        self.message = ChatMessage(role=MessageRole.ASSISTANT, content=content)


class _EchoLILLM:
    """Fake LlamaIndex LLM: records `.chat` calls, returns a canned response."""

    def __init__(self, content: str):
        self._content = content
        self.chat_calls: list[list[Any]] = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append(list(messages))
        return _EchoLIResponse(self._content)


@pytest.fixture
def echo_llamaindex_llm(monkeypatch):
    """Patch `_get_llm()` in the LlamaIndex service module.

    Also clears the ChatMemoryBuffer singleton so history doesn't leak
    across tests, mirroring the LangChain fixture.
    """
    llm = _EchoLILLM("respuesta-eco-li")

    from app.services.llamaindex_rag import db_comm as _db
    _db._get_llm.cache_clear()
    monkeypatch.setattr(
        "app.services.llamaindex_rag.service._get_llm", lambda: llm
    )
    monkeypatch.setattr(
        "app.services.llamaindex_rag.db_comm._get_llm", lambda: llm
    )

    from app.services.llamaindex_rag.memory import clear_history
    clear_history()

    yield llm

    clear_history()
    # No cache_clear on teardown: monkeypatch restores the original lru_cache
    # function automatically, so an explicit cache_clear here would hit the
    # (already-replaced) lambda and fail.


# ---------- Reranker mock (shared by both engines) ----------

@pytest.fixture
def mock_reranker(monkeypatch):
    """Patch httpx.post used by utils.rerank_texts. Scores = 1/(i+1) keeping order."""
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


# ---------- PDF loader mocks ----------

@pytest.fixture
def mock_pypdf_loader(monkeypatch):
    """Patch LangChain's PyPDFLoader so it returns canned pages indexed by filename."""
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


@pytest.fixture
def mock_li_pdf_reader(monkeypatch):
    """Patch LlamaIndex's PDFReader so it returns canned pages indexed by filename.

    Returns fresh Document instances on every call: the real `PDFReader` reads
    from disk each time and the service mutates metadata in place, so reusing
    the same objects would leak state across `_ingest` invocations.
    """
    from llama_index.core.schema import Document as LIDocument

    pages_by_name: dict[str, list[LIDocument]] = {}

    def _clone_docs(docs: list[LIDocument]) -> list[LIDocument]:
        return [LIDocument(text=d.text, metadata=dict(d.metadata)) for d in docs]

    class _FakeReader:
        def load_data(self, file, extra_info=None, fs=None):
            name = Path(file).name
            if name in pages_by_name:
                return _clone_docs(pages_by_name[name])
            return [
                LIDocument(
                    text=f"page 0 of {name}",
                    metadata={"page_label": "1", "file_name": name},
                ),
                LIDocument(
                    text=f"page 1 of {name}",
                    metadata={"page_label": "2", "file_name": name},
                ),
            ]

    monkeypatch.setattr(
        "app.services.llamaindex_rag.service.PDFReader", lambda: _FakeReader()
    )
    return pages_by_name


# ---------- Real ChromaDB test collection (LangChain) ----------

@pytest.fixture(scope="session")
def chroma_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@pytest.fixture(scope="session")
def test_collection_seeded(chroma_client):
    """Create langchain_rag_test as a copy of langchain_rag (session-scoped).

    Destroyed at the end of the session so langchain_rag stays intact.
    """
    try:
        chroma_client.delete_collection(TEST_COLLECTION_NAME)
    except Exception:
        pass

    test_col = chroma_client.create_collection(
        name=TEST_COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

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


# ---------- Real ChromaDB test collection (LlamaIndex) ----------

@pytest.fixture(scope="session")
def li_test_collection_seeded(chroma_client):
    """Create llamaindex_rag_test as a copy of llamaindex_rag (session-scoped)."""
    try:
        chroma_client.delete_collection(LI_TEST_COLLECTION_NAME)
    except Exception:
        pass

    test_col = chroma_client.create_collection(
        name=LI_TEST_COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

    try:
        src = chroma_client.get_collection(LI_COLLECTION_NAME)
        data = src.get(include=["documents", "metadatas", "embeddings"])
        if data["ids"]:
            test_col.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"],
            )
    except Exception:
        pass

    yield test_col

    try:
        chroma_client.delete_collection(LI_TEST_COLLECTION_NAME)
    except Exception:
        pass


@pytest.fixture
def patched_llamaindex_db(monkeypatch, chroma_client, li_test_collection_seeded):
    """Redirect LlamaindexSrv to the llamaindex_rag_test collection.

    We patch the native Chroma collection factory (`_get_chroma_collection`)
    and rebuild the `ChromaVectorStore` / index singletons so the service
    talks to the test collection instead of the production one.
    """
    from app.services.llamaindex_rag import db_comm as _db
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex

    test_collection = li_test_collection_seeded
    test_vector_store = ChromaVectorStore(chroma_collection=test_collection)
    test_storage = StorageContext.from_defaults(vector_store=test_vector_store)
    test_index = VectorStoreIndex.from_vector_store(
        vector_store=test_vector_store,
        storage_context=test_storage,
        embed_model=_db._get_embed_model(),
    )

    # Patch the service-side lookups so the whole flow uses the test objects.
    monkeypatch.setattr(
        "app.services.llamaindex_rag.service._get_chroma_collection",
        lambda: test_collection,
    )
    monkeypatch.setattr(
        "app.services.llamaindex_rag.service.get_index",
        lambda: test_index,
    )
    monkeypatch.setattr(
        "app.services.llamaindex_rag.service.get_collection_count",
        lambda: test_collection.count(),
    )
    return test_collection


@pytest.fixture
def empty_li_test_collection(chroma_client, li_test_collection_seeded):
    """llamaindex_rag_test emptied for ingest tests; re-seeded on teardown."""
    all_ids = li_test_collection_seeded.get()["ids"]
    if all_ids:
        li_test_collection_seeded.delete(ids=all_ids)
    yield li_test_collection_seeded
    try:
        src = chroma_client.get_collection(LI_COLLECTION_NAME)
        data = src.get(include=["documents", "metadatas", "embeddings"])
        if data["ids"]:
            current = li_test_collection_seeded.get()["ids"]
            if current:
                li_test_collection_seeded.delete(ids=current)
            li_test_collection_seeded.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"],
            )
    except Exception:
        pass
