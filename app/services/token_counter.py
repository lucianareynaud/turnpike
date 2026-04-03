"""BPE token counting for context window management.

Single source of truth for token counting in the application layer.
tiktoken must not be imported anywhere else in the codebase.
"""

import functools

import tiktoken


class ContextTooLargeError(ValueError):
    """Raised when an assembled context exceeds the configured token limit.

    Subclasses ValueError so it is treated as a bad-input condition rather
    than an internal server error. Carries structured attributes so the
    HTTP layer can include exact counts in the 400 response body.
    """

    def __init__(self, actual_tokens: int, max_tokens: int) -> None:
        self.actual_tokens = actual_tokens
        self.max_tokens = max_tokens
        super().__init__(f"Context too large: {actual_tokens} tokens exceeds limit of {max_tokens}")


@functools.lru_cache(maxsize=16)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding for `model`, cached per model name.

    Falls back to cl100k_base for unknown model identifiers.
    The fallback is intentionally conservative: returning a valid approximate
    token count is preferable to raising.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the BPE token count for `text` using the vocabulary for `model`.

    This counts the final assembled string used by the current application
    layer. It is not a canonical provider billing/accounting primitive.

    Args:
        text: Input string to tokenize.
        model: Model identifier used to select the tokenizer vocabulary.
            Falls back to cl100k_base for unknown identifiers.

    Returns:
        Number of tokens in `text`. Returns 0 for empty strings.
    """
    if not text:
        return 0
    return len(_get_encoding(model).encode(text))
