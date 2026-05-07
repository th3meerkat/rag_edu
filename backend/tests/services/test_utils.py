"""Regression tests for app.services.utils (framework-agnostic)."""
from __future__ import annotations

import json

import pytest

from app.services.utils import (
    build_filter,
    detect_positional,
    load_manifest,
    manifest_path,
    rerank_texts,
    rrf_fuse,
    save_manifest,
)


# ---------- Manifest I/O (per-engine) ----------

class TestManifest:
    def test_load_missing_returns_empty(self, temp_manifest_path):
        # temp_manifest_path points at .../ingested_test.json
        assert not temp_manifest_path.exists()
        assert load_manifest("test") == {}

    def test_save_then_load_round_trip(self, temp_manifest_path, fake_manifest):
        save_manifest("test", fake_manifest)
        assert temp_manifest_path.exists()
        assert load_manifest("test") == fake_manifest

    def test_save_is_valid_json_utf8(self, temp_manifest_path):
        save_manifest("test", {"año.pdf": 3})
        raw = temp_manifest_path.read_text(encoding="utf-8")
        assert "año.pdf" in raw  # ensure_ascii=False preserves tildes
        assert json.loads(raw) == {"año.pdf": 3}

    def test_per_engine_paths_are_isolated(self, manifest_dir):
        """langchain and llamaindex manifests live in separate files."""
        save_manifest("langchain", {"a.pdf": 1})
        save_manifest("llamaindex", {"b.pdf": 2})
        assert load_manifest("langchain") == {"a.pdf": 1}
        assert load_manifest("llamaindex") == {"b.pdf": 2}
        assert manifest_path("langchain") != manifest_path("llamaindex")


# ---------- Positional detection ----------

class TestDetectPositional:
    @pytest.mark.parametrize(
        "msg,expected",
        [
            ("¿qué pasa en la página 5?", ("pagina", 5)),
            ("dime algo de la Pagina 12", ("pagina", 12)),
            ("resumen del capítulo 3", ("capitulo", 3)),
            ("Capitulo 1 por favor", ("capitulo", 1)),
            ("¿qué ocurre al final del libro?", ("final", None)),
            ("quiero la última página", ("final", None)),
            ("cuenta el desenlace", ("final", None)),
            ("al principio del libro", ("principio", None)),
            ("la primera página del libro", ("principio", None)),
            ("al comienzo del cuento", ("principio", None)),
        ],
    )
    def test_detects_patterns(self, msg, expected):
        assert detect_positional(msg) == expected

    def test_specific_pattern_wins_over_generic(self):
        assert detect_positional("en la página 5 hay un final bonito") == ("pagina", 5)

    def test_no_pattern(self):
        assert detect_positional("hola, ¿de qué trata el libro?") is None


# ---------- Filter building ----------

class TestBuildFilter:
    def test_pagina_is_zero_indexed(self, fake_manifest):
        where, where_doc = build_filter(("pagina", 5), fake_manifest)
        assert where == {"page": 4}
        assert where_doc is None

    def test_capitulo_uses_document_contains(self, fake_manifest):
        where, where_doc = build_filter(("capitulo", 3), fake_manifest)
        assert where is None
        assert where_doc == {"$contains": "Capítulo 3"}

    def test_final_single_source(self):
        manifest = {"book.pdf": 100}
        where, where_doc = build_filter(("final", None), manifest)
        assert where_doc is None
        assert where == {
            "$and": [
                {"source": "book.pdf"},
                {"page": {"$gte": 90}},
                {"page": {"$lte": 99}},
            ]
        }

    def test_principio_single_source(self):
        manifest = {"book.pdf": 100}
        where, _ = build_filter(("principio", None), manifest)
        assert where == {
            "$and": [
                {"source": "book.pdf"},
                {"page": {"$gte": 0}},
                {"page": {"$lte": 9}},
            ]
        }

    def test_final_multiple_sources_uses_or(self, fake_manifest):
        where, _ = build_filter(("final", None), fake_manifest)
        assert "$or" in where
        assert len(where["$or"]) == 2

    def test_empty_manifest_returns_none(self):
        assert build_filter(("final", None), {}) == (None, None)

    def test_small_manifest_window_is_at_least_one(self):
        where, _ = build_filter(("final", None), {"tiny.pdf": 3})
        assert where == {
            "$and": [
                {"source": "tiny.pdf"},
                {"page": {"$gte": 2}},
                {"page": {"$lte": 2}},
            ]
        }

    def test_unknown_kind_returns_none(self):
        assert build_filter(("desconocido", None), {}) == (None, None)


# ---------- Generic RRF fusion ----------

class TestRrfFuse:
    def test_duplicate_across_lists_is_fused(self):
        # Generic: fuse lists of strings (key_fn = identity).
        result = rrf_fuse(
            [["same", "other"], ["same"]],
            key_fn=lambda s: s,
            k=60,
            top_n=10,
        )
        assert len(result) == 2
        top_scores = [s for s, _ in result]
        assert top_scores[0] > top_scores[1]
        assert result[0][1] == "same"

    def test_empty_input(self):
        assert rrf_fuse([], key_fn=lambda x: x, k=60, top_n=5) == []

    def test_top_n_limits_output(self):
        items = [f"d{i}" for i in range(5)]
        result = rrf_fuse([items], key_fn=lambda s: s, k=60, top_n=3)
        assert len(result) == 3

    def test_score_matches_formula(self):
        result = rrf_fuse([["x"]], key_fn=lambda s: s, k=60, top_n=1)
        assert result[0][0] == pytest.approx(1.0 / (60 + 1))


# ---------- Generic reranker HTTP call ----------

class TestRerankTexts:
    def test_empty_texts_short_circuit(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "app.services.utils.httpx.post",
            lambda *a, **kw: calls.append(1),
        )
        assert rerank_texts("q", [], top_n=3) == []
        assert calls == []

    def test_sorts_by_score_and_limits_top_n(self, mock_reranker):
        texts = [f"d{i}" for i in range(5)]
        # mock_reranker assigns score 1/(i+1): d0>d1>d2>d3>d4
        result = rerank_texts("q", texts, top_n=3)
        assert len(result) == 3
        assert [i for i, _ in result] == [0, 1, 2]
        assert [s for _, s in result] == [1.0, 0.5, pytest.approx(1 / 3)]

    def test_request_shape(self, mock_reranker):
        from app.config import RERANKER_MODEL
        rerank_texts("pregunta", ["hello"], top_n=1)
        assert len(mock_reranker) == 1
        call = mock_reranker[0]
        assert call["url"].endswith("/rerank")
        assert call["json"]["query"] == "pregunta"
        assert call["json"]["documents"] == ["hello"]
        assert call["json"]["model"] == RERANKER_MODEL
