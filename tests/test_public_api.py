"""Smoke test: verify that the top-level public API surface matches docs."""

import turnpike

EXPECTED_EXPORTS = {
    "__version__",
    "LLMRequestEnvelope",
    "LLMRequestContext",
    "EnvelopeStatus",
    "CostSource",
    "GatewayResult",
    "GatewayStream",
    "StreamChunk",
    "call_llm",
    "call_llm_stream",
    "ProviderBase",
    "ProviderResponse",
    "register_provider",
    "RoutePolicy",
    "register_route_policy",
    "estimate_cost",
    "register_pricing",
    "emit_event",
    "setup_otel",
    "shutdown_otel",
}


def test_all_matches_expected():
    assert set(turnpike.__all__) == EXPECTED_EXPORTS


def test_exports_are_importable():
    for name in EXPECTED_EXPORTS:
        assert hasattr(turnpike, name), f"turnpike.{name} not importable"


def test_version_is_string():
    assert isinstance(turnpike.__version__, str)
    assert len(turnpike.__version__) > 0
