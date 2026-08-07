"""Bound in-memory Streamlit session data for this local learning tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from utils.usage import safe_session_record

MAX_SESSION_RECORDS = 100
MAX_CHAT_HISTORY_MESSAGES = 20
MAX_CHAT_HISTORY_CHARACTERS = 24_000


def append_bounded_record(
    records: list[dict[str, Any]], record: Mapping[str, Any]
) -> None:
    records.append(safe_session_record(record))
    del records[:-MAX_SESSION_RECORDS]


def bounded_chat_history(
    history: Sequence[Mapping[str, str]],
    new_messages: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    """Keep the newest valid messages within count and character limits."""
    candidates: list[dict[str, str]] = []
    for message in (*history, *new_messages):
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            candidates.append({"role": role, "content": content})

    kept_reversed: list[dict[str, str]] = []
    remaining = MAX_CHAT_HISTORY_CHARACTERS
    for message in reversed(candidates[-MAX_CHAT_HISTORY_MESSAGES:]):
        if remaining <= 0:
            break
        content = message["content"]
        if len(content) > remaining:
            content = content[:remaining]
        kept_reversed.append({"role": message["role"], "content": content})
        remaining -= len(content)
    return list(reversed(kept_reversed))
