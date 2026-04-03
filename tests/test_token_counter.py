"""Tests for token counting module."""

from app.services.token_counter import _get_encoding, count_tokens


def test_known_string_exact_count():
    # "Hello world" → tiktoken gpt-4o → 2 tokens
    assert count_tokens("Hello world", "gpt-4o") == 2


def test_empty_string_returns_zero():
    assert count_tokens("", "gpt-4o") == 0


def test_unknown_model_falls_back():
    # This test verifies only that the fallback path is stable: it does not
    # raise and returns a positive integer. It does NOT assert semantic
    # equivalence of token counts between the fallback encoder and a known
    # model — different vocabularies produce different counts by design.
    # Do not "improve" this test by pinning an exact fallback count.
    known = count_tokens("test", "gpt-4o-mini")
    fallback = count_tokens("test", "nonexistent-model-xyz-999")
    assert isinstance(fallback, int)
    assert fallback > 0
    assert fallback <= known * 3


def test_encoder_cached():
    _get_encoding.cache_clear()
    count_tokens("a", "gpt-4o")
    count_tokens("a", "gpt-4o")
    assert _get_encoding.cache_info().hits >= 1
