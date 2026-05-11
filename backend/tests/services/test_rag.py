"""Regression tests for app.services.rag.RagService (template method).

`query()` became abstract in the refactor (each engine implements it natively),
so these tests cover only the concrete template method that survives at the
base-class level: `run_ingestion`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from app.services.rag import RagService


class _FakeRagService(RagService):
    """Concrete implementation used only to exercise the template methods."""

    engine_name: ClassVar[str] = "test"

    def __init__(self, ingest_result: dict[str, int] | None = None):
        self.ingest_calls: list[list[Path]] = []
        self._ingest_result = ingest_result or {}

    def _ingest(self, pdf_paths):
        self.ingest_calls.append(list(pdf_paths))
        return self._ingest_result

    def _retrieve(self, query, where=None, where_document=None):
        return []

    def _generate(self, question, docs):
        return f"answer::{question}"

    def query(self, msg: str) -> str:  # pragma: no cover — not exercised here
        raise NotImplementedError


class TestRunIngestion:
    def test_no_pdfs_short_circuits(
        self, monkeypatch, tmp_path, temp_manifest_path
    ):
        monkeypatch.setattr("app.services.rag.RAW_DATA_DIR", tmp_path)
        srv = _FakeRagService()
        srv.run_ingestion()
        assert srv.ingest_calls == []
        assert not temp_manifest_path.exists()

    def test_all_pdfs_already_ingested(
        self, monkeypatch, tmp_path, temp_manifest_path
    ):
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.0")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.0")
        temp_manifest_path.write_text(json.dumps({"a.pdf": 3, "b.pdf": 4}))
        monkeypatch.setattr("app.services.rag.RAW_DATA_DIR", tmp_path)

        srv = _FakeRagService()
        srv.run_ingestion()
        assert srv.ingest_calls == []
        assert json.loads(temp_manifest_path.read_text()) == {
            "a.pdf": 3,
            "b.pdf": 4,
        }

    def test_new_pdfs_are_ingested_and_manifest_updated(
        self, monkeypatch, tmp_path, temp_manifest_path
    ):
        (tmp_path / "old.pdf").write_bytes(b"%PDF-1.0")
        (tmp_path / "new.pdf").write_bytes(b"%PDF-1.0")
        temp_manifest_path.write_text(json.dumps({"old.pdf": 2}))
        monkeypatch.setattr("app.services.rag.RAW_DATA_DIR", tmp_path)

        srv = _FakeRagService(ingest_result={"new.pdf": 7})
        srv.run_ingestion()

        assert len(srv.ingest_calls) == 1
        ingested_names = [p.name for p in srv.ingest_calls[0]]
        assert ingested_names == ["new.pdf"]
        assert json.loads(temp_manifest_path.read_text()) == {
            "old.pdf": 2,
            "new.pdf": 7,
        }
