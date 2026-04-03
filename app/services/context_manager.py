"""Context manager service for conversation-turn context preparation.

This module applies bounded context strategies for multi-turn conversations.
It supports:
- full
- sliding_window
- summarized

All strategies are deterministic and reproducible in the MVP.
"""

import os
from typing import Literal

from app.services.token_counter import ContextTooLargeError, count_tokens

ContextStrategy = Literal["full", "sliding_window", "summarized"]

SLIDING_WINDOW_SIZE = 5
SUMMARIZATION_THRESHOLD = 10


def prepare_context(
    history: list[str],
    message: str,
    strategy: ContextStrategy,
    model: str = "gpt-4o",
) -> tuple[str, int]:
    """Prepare conversation context according to the requested strategy.

    Args:
        history: Previous conversation messages in chronological order.
        message: Current user message.
        strategy: Context preparation strategy.
        model: Model identifier used only for token counting in this layer.
            This is not the authoritative routing decision for the gateway.

    Returns:
        A tuple of:
        - prepared_context: formatted context string
        - token_count: exact BPE token count for the prepared context
    """
    if strategy == "full":
        context = _prepare_full_context(history, message)

    elif strategy == "sliding_window":
        context = _prepare_sliding_window_context(history, message)

    elif strategy == "summarized":
        context = _prepare_summarized_context(history, message)

    else:
        raise ValueError(f"Unknown context strategy: {strategy}")

    max_ctx = int(os.environ.get("MAX_CONTEXT_TOKENS", "8192"))
    total_tokens = count_tokens(context, model)
    if total_tokens > max_ctx:
        raise ContextTooLargeError(actual_tokens=total_tokens, max_tokens=max_ctx)

    return context, total_tokens


def _prepare_full_context(history: list[str], message: str) -> str:
    """Include all prior history plus the current message."""
    messages = history + [message]
    return _format_messages(messages)


def _prepare_sliding_window_context(history: list[str], message: str) -> str:
    """Include only the most recent bounded portion of history plus the current message."""
    recent_history = history[-SLIDING_WINDOW_SIZE:]
    messages = recent_history + [message]
    return _format_messages(messages)


def _prepare_summarized_context(history: list[str], message: str) -> str:
    """Summarize older history deterministically and keep recent turns verbatim.

    In the MVP, summarization is a deterministic placeholder string.
    If a future implementation uses an LLM for summarization, that call
    must go through the gateway layer.
    """
    if len(history) <= SUMMARIZATION_THRESHOLD:
        return _prepare_full_context(history, message)

    old_history = history[:-SLIDING_WINDOW_SIZE]
    recent_history = history[-SLIDING_WINDOW_SIZE:]

    summary = _build_placeholder_summary(old_history)

    context_parts = [summary, _format_messages(recent_history), f"Current: {message}"]
    return "\n".join(part for part in context_parts if part)


def _format_messages(messages: list[str]) -> str:
    """Format messages into a deterministic multi-line context block."""
    return "\n".join(f"Turn {index}: {content}" for index, content in enumerate(messages))


def _build_placeholder_summary(old_history: list[str]) -> str:
    """Build a deterministic placeholder summary for older history."""
    return (
        f"[Summary of {len(old_history)} earlier messages: "
        "prior conversation context retained in condensed form]"
    )
