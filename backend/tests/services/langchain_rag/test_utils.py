"""Regression tests for LangChain-specific helpers.

`expand_queries` and `rerank` moved here from the shared utils module; their
contract is unchanged so the assertions mirror the previous tests.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from app.services.langchain_rag.utils import (
    expand_queries,
    format_docs,
    rerank,
    rrf_fuse_docs,
    sanitize_question,
)


class TestExpandQueries:
    def test_happy_path(self, echo_chat_openai):
        queries = expand_queries("¿qué come el principito?")
        assert queries == ["q-alt-1", "q-alt-2"]

    def test_raises_when_fewer_than_expected(self, monkeypatch):
        class _BadLLM:
            def invoke(self, _):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({"queries": ["only-one"]}))

        monkeypatch.setattr(
            "app.services.langchain_rag.utils.ChatOpenAI", lambda *a, **kw: _BadLLM()
        )
        with pytest.raises(ValueError, match="Expansion returned"):
            expand_queries("anything")

    def test_truncates_if_more_than_expected(self, monkeypatch):
        class _BigLLM:
            def invoke(self, _):
                from langchain_core.messages import AIMessage
                return AIMessage(
                    content=json.dumps({"queries": ["a", "b", "c", "d"]})
                )

        monkeypatch.setattr(
            "app.services.langchain_rag.utils.ChatOpenAI", lambda *a, **kw: _BigLLM()
        )
        assert expand_queries("x") == ["a", "b"]


class TestRerank:
    def _doc(self, text: str) -> Document:
        return Document(page_content=text, metadata={})

    def test_empty_docs_short_circuit(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "app.services.utils.httpx.post",
            lambda *a, **kw: calls.append(1),
        )
        assert rerank("q", [], top_n=3) == []
        assert calls == []

    def test_sorts_by_score_and_limits_top_n(self, mock_reranker):
        docs = [self._doc(f"d{i}") for i in range(5)]
        result = rerank("q", docs, top_n=3)
        assert len(result) == 3
        assert [d.page_content for d, _ in result] == ["d0", "d1", "d2"]
        assert [s for _, s in result] == [1.0, 0.5, pytest.approx(1 / 3)]


class TestRrfFuseDocs:
    def _doc(self, text: str) -> Document:
        return Document(page_content=text, metadata={})

    def test_dedupes_by_page_content(self):
        a = self._doc("same")
        b = self._doc("same")
        c = self._doc("other")
        result = rrf_fuse_docs([[a, c], [b]], top_n=10)
        assert len(result) == 2
        assert result[0][1].page_content == "same"


class TestFormatDocs:
    def test_empty_docs_emits_sentinel(self):
        assert format_docs([]) == "(sin contexto)"

    def test_renders_each_doc_with_metadata(self):
        doc = Document(
            page_content="hola",
            metadata={"source": "book.pdf", "page": 2},
        )
        rendered = format_docs([doc])
        assert "source=book.pdf" in rendered
        assert "page=2" in rendered
        assert "hola" in rendered


class TestSanitizeQuestion:
    def test_strips_control_chars(self):
        assert "\x00" not in sanitize_question("a\x00b")

    def test_caps_length(self):
        s = "a" * 5000
        assert len(sanitize_question(s)) == 2000
