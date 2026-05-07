"""Regression tests for LlamaIndex-specific helpers."""
from __future__ import annotations

import json

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.llamaindex_rag.utils import (
    build_chunk_ids,
    expand_queries,
    format_nodes,
    rrf_fuse_nodes,
    sanitize_question,
)


class _FakeResponse:
    def __init__(self, content: str):
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        self.message = ChatMessage(role=MessageRole.ASSISTANT, content=content)


class _FakeLILLM:
    def __init__(self, content: str):
        self._content = content
        self.chat_calls: list = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        return _FakeResponse(self._content)


class TestExpandQueries:
    def test_happy_path(self, monkeypatch):
        llm = _FakeLILLM(json.dumps({"queries": ["q-alt-1", "q-alt-2"]}))
        monkeypatch.setattr(
            "app.services.llamaindex_rag.utils.LIOpenAI",
            lambda *a, **kw: llm,
        )
        assert expand_queries("¿qué come el principito?") == ["q-alt-1", "q-alt-2"]

    def test_raises_when_fewer_than_expected(self, monkeypatch):
        llm = _FakeLILLM(json.dumps({"queries": ["only-one"]}))
        monkeypatch.setattr(
            "app.services.llamaindex_rag.utils.LIOpenAI",
            lambda *a, **kw: llm,
        )
        with pytest.raises(ValueError, match="Expansion returned"):
            expand_queries("anything")

    def test_truncates_if_more_than_expected(self, monkeypatch):
        llm = _FakeLILLM(json.dumps({"queries": ["a", "b", "c", "d"]}))
        monkeypatch.setattr(
            "app.services.llamaindex_rag.utils.LIOpenAI",
            lambda *a, **kw: llm,
        )
        assert expand_queries("x") == ["a", "b"]


def _node(text: str, **meta) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata=meta), score=1.0)


class TestRrfFuseNodes:
    def test_dedupes_by_node_text(self):
        a = _node("same")
        b = _node("same")
        c = _node("other")
        result = rrf_fuse_nodes([[a, c], [b]], top_n=10)
        assert len(result) == 2
        assert result[0][1].node.get_content() == "same"


class TestFormatNodes:
    def test_empty_emits_sentinel(self):
        assert format_nodes([]) == "(sin contexto)"

    def test_renders_each_node_with_metadata(self):
        n = _node("hola", source="book.pdf", page=2)
        rendered = format_nodes([n])
        assert "source=book.pdf" in rendered
        assert "page=2" in rendered
        assert "hola" in rendered


class TestSanitizeQuestion:
    def test_strips_control_chars(self):
        assert "\x00" not in sanitize_question("a\x00b")

    def test_caps_length(self):
        s = "a" * 5000
        assert len(sanitize_question(s)) == 2000


class TestBuildChunkIds:
    def test_ids_are_deterministic_and_per_page_scoped(self):
        nodes = [
            TextNode(text="a", metadata={"source": "x.pdf", "page": 0}),
            TextNode(text="b", metadata={"source": "x.pdf", "page": 0}),
            TextNode(text="c", metadata={"source": "x.pdf", "page": 1}),
        ]
        ids = build_chunk_ids(nodes)
        assert ids == ["x.pdf:0:0", "x.pdf:0:1", "x.pdf:1:0"]
