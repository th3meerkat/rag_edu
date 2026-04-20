"""Regression tests for app.services.rag.RagService (template method)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.rag import RagService


class _FakeRagService(RagService):
    """Concrete implementation used only to exercise the template methods."""

    def __init__(self, ingest_result: dict[str, int] | None = None):
        self.ingest_calls: list[list[Path]] = []
        self.retrieve_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self._ingest_result = ingest_result or {}

    def _ingest(self, pdf_paths):
        self.ingest_calls.append(list(pdf_paths))
        return self._ingest_result

    def _retrieve(self, query, where=None, where_document=None):
        self.retrieve_calls.append(
            {"query": query, "where": where, "where_document": where_document}
        )
        return [Document(page_content=f"doc-for-{query}", metadata={})]

    def _generate(self, question, docs):
        self.generate_calls.append({"question": question, "docs": docs})
        return f"answer::{question}"


@pytest.fixture
def no_dotenv(monkeypatch):
    """Prevent load_dotenv from touching the real env file."""
    monkeypatch.setattr("app.services.rag.load_dotenv", lambda *a, **kw: None)


@pytest.fixture
def stub_rerank(monkeypatch):
    """Stub rerank inside rag.py to a deterministic pass-through."""
    monkeypatch.setattr(
        "app.services.rag.rerank",
        lambda query, docs, top_n: [(d, 1.0 - i * 0.1) for i, d in enumerate(docs)],
    )


class TestQuery:
    def test_no_positional_pattern_skips_filter(self, no_dotenv, stub_rerank):
        srv = _FakeRagService()
        answer = srv.query("¿de qué trata el libro?")
        assert answer == "answer::¿de qué trata el libro?"
        # Single-query pipeline (expand disabled in _expand_queries).
        assert len(srv.retrieve_calls) == 1
        call = srv.retrieve_calls[0]
        assert call["where"] is None
        assert call["where_document"] is None

    def test_positional_pattern_applies_filter(
        self, no_dotenv, stub_rerank, temp_manifest_path
    ):
        temp_manifest_path.write_text(json.dumps({"book.pdf": 100}))
        srv = _FakeRagService()
        srv.query("¿qué pasa en la página 5?")
        call = srv.retrieve_calls[0]
        # pagina N: user says 5 → zero-indexed 4
        assert call["where"] == {"page": 4}

    def test_generate_receives_reranked_docs_only(self, no_dotenv, stub_rerank):
        srv = _FakeRagService()
        srv.query("hola")
        assert len(srv.generate_calls) == 1
        docs = srv.generate_calls[0]["docs"]
        assert all(isinstance(d, Document) for d in docs)


class TestRunIngestion:
    def test_no_pdfs_short_circuits(
        self, no_dotenv, monkeypatch, tmp_path, temp_manifest_path
    ):
        monkeypatch.setattr("app.services.rag.DATA_DIR", tmp_path)
        srv = _FakeRagService()
        srv.run_ingestion()
        assert srv.ingest_calls == []
        assert not temp_manifest_path.exists()

    def test_all_pdfs_already_ingested(
        self, no_dotenv, monkeypatch, tmp_path, temp_manifest_path
    ):
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.0")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.0")
        temp_manifest_path.write_text(json.dumps({"a.pdf": 3, "b.pdf": 4}))
        monkeypatch.setattr("app.services.rag.DATA_DIR", tmp_path)

        srv = _FakeRagService()
        srv.run_ingestion()
        assert srv.ingest_calls == []
        # Manifest stays untouched.
        assert json.loads(temp_manifest_path.read_text()) == {
            "a.pdf": 3,
            "b.pdf": 4,
        }

    def test_new_pdfs_are_ingested_and_manifest_updated(
        self, no_dotenv, monkeypatch, tmp_path, temp_manifest_path
    ):
        (tmp_path / "old.pdf").write_bytes(b"%PDF-1.0")
        (tmp_path / "new.pdf").write_bytes(b"%PDF-1.0")
        temp_manifest_path.write_text(json.dumps({"old.pdf": 2}))
        monkeypatch.setattr("app.services.rag.DATA_DIR", tmp_path)

        srv = _FakeRagService(ingest_result={"new.pdf": 7})
        srv.run_ingestion()

        assert len(srv.ingest_calls) == 1
        ingested_names = [p.name for p in srv.ingest_calls[0]]
        assert ingested_names == ["new.pdf"]
        assert json.loads(temp_manifest_path.read_text()) == {
            "old.pdf": 2,
            "new.pdf": 7,
        }
