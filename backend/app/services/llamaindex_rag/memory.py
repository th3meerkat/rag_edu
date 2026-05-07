"""Short-term conversational memory for the LlamaIndex RAG service.

Mirrors the LangChain counterpart: ephemeral, process-wide, single global
conversation, windowed to the last 5 turns. We reuse LlamaIndex's native
`ChatMemoryBuffer` as the underlying store (so the types we produce and
consume are first-class `ChatMessage` objects the LlamaIndex chat APIs
already understand) and add a *message-count* window on top of it at read
time — `ChatMemoryBuffer` itself is token-limited, which does not match the
"last N turns" spec the project asked for.
"""
from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

# Window size. 5 turns = 10 messages (1 user + 1 assistant per turn).
MAX_TURNS = 5
MAX_MESSAGES = MAX_TURNS * 2

# Effectively disable the built-in token ceiling: our window is message-based.
# (Large enough that the buffer never drops messages on its own.)
_UNBOUNDED_TOKENS = 10**9

_memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(token_limit=_UNBOUNDED_TOKENS)


def recent_messages() -> list[ChatMessage]:
    """Return at most the last `MAX_MESSAGES` messages stored.

    Called when assembling the prompt, so the LLM sees only the recent
    window, even though the buffer keeps the full history in memory.
    """
    all_msgs = _memory.get_all()
    return all_msgs[-MAX_MESSAGES:]


def append_turn(user_msg: str, assistant_reply: str) -> None:
    """Persist one (user, assistant) turn after a successful generation."""
    _memory.put(ChatMessage(role=MessageRole.USER, content=user_msg))
    _memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_reply))


def clear_history() -> None:
    """Wipe the current conversation (useful for tests)."""
    _memory.reset()
