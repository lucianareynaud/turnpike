"""Turnpike — typed envelope and telemetry primitives for LLM cost attribution.

Public API surface. Import from here for stable access.
Internal module paths (turnpike.gateway.client, etc.) are implementation
details and may change between minor versions.

Vendor-specific providers (OpenAIProvider, AnthropicProvider) are available
from ``turnpike.gateway.provider`` but are not re-exported here because
they depend on optional SDKs and may track upstream API changes.
"""

from importlib.metadata import version as _pkg_version

from turnpike.context import LLMRequestContext
from turnpike.envelope import (
    CostSource,
    EnvelopeStatus,
    LLMRequestEnvelope,
)
from turnpike.gateway.client import (
    GatewayResult,
    GatewayStream,
    StreamChunk,
    call_llm,
    call_llm_stream,
)
from turnpike.gateway.cost_model import estimate_cost, register_pricing
from turnpike.gateway.otel_setup import setup_otel, shutdown_otel
from turnpike.gateway.policies import (
    RoutePolicy,
    register_route_policy,
)
from turnpike.gateway.provider import (
    ProviderBase,
    ProviderResponse,
    register_provider,
)
from turnpike.gateway.telemetry import emit_event

__version__ = _pkg_version("turnpike")

__all__ = [
    # Version
    "__version__",
    # Envelope and context
    "LLMRequestEnvelope",
    "LLMRequestContext",
    "EnvelopeStatus",
    "CostSource",
    # Gateway results
    "GatewayResult",
    "GatewayStream",
    "StreamChunk",
    # Call entrypoints
    "call_llm",
    "call_llm_stream",
    # Provider abstraction
    "ProviderBase",
    "ProviderResponse",
    "register_provider",
    # Policy
    "RoutePolicy",
    "register_route_policy",
    # Cost
    "estimate_cost",
    "register_pricing",
    # Telemetry
    "emit_event",
    "setup_otel",
    "shutdown_otel",
]
