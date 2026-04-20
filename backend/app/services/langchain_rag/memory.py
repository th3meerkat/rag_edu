"""Short-term conversational memory for the Langchain RAG service.

Design choices (all driven by the project requirements):
  - Ephemeral: lives in the process; a restart wipes it. No persistence layer.
  - No session identity: a single global conversation. `RunnableWithMessageHistory`
    still requires a session_id internally, so we pass a fixed constant.
  - Window of the last 5 turns (= 10 messages: 5 human + 5 ai). Trimming
    happens at prompt-assembly time via `trim_messages`, leaving the store
    itself untouched — the idiomatic Langchain v1 pattern.
"""
from operator import itemgetter

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import trim_messages
from langchain_core.runnables import Runnable, RunnablePassthrough

# Fixed session id: `RunnableWithMessageHistory` is keyed by session_id, but we
# expose a single "current conversation" at process level, so any constant works.
SESSION_ID = "default"

# Window size. 5 turns = 10 messages (1 human + 1 ai per turn).
MAX_TURNS = 5
MAX_MESSAGES = MAX_TURNS * 2

# Process-wide singleton. InMemoryChatMessageHistory is thread-safe enough for
# our use (no concurrent same-session requests expected).
_history = InMemoryChatMessageHistory()


def get_session_history(session_id: str) -> BaseChatMessageHistory:  # noqa: ARG001
    """Factory required by `RunnableWithMessageHistory`.

    session_id is ignored: we return the same global history regardless. Keeps
    the API surface compatible with Langchain's expectation while matching
    the "no sessions" requirement.
    """
    return _history


def clear_history() -> None:
    """Wipe the current conversation (useful for tests)."""
    _history.clear()


def build_history_trimmer() -> Runnable:
    """Runnable that trims the `history` key to the last MAX_MESSAGES messages.

    Applied as a `RunnablePassthrough.assign` step before the prompt template
    consumes the placeholder. `token_counter=len` means "count messages, not
    tokens" — the user asked for a turn-based window, not a token budget.
    `strategy="last"` keeps the newest; `start_on="human"` guarantees the
    trimmed sequence starts with a human turn so the prompt stays well-formed.
    """
    trimmer = trim_messages(
        max_tokens=MAX_MESSAGES,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=False,
    )
    return RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
