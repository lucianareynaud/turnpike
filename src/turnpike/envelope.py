"""
Turnpike envelope — the portable, versioned contract for LLM request lifecycle.

This module defines the typed envelope primitives that can be used across
gateway, SDK, and sidecar runtimes without framework dependencies.
"""

from dataclasses import dataclass, field, fields
from enum import Enum, StrEnum
from typing import Any

ENVELOPE_SCHEMA_VERSION = "0.1.0"

__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "EnvelopeStatus",
    "CostSource",
    "CircuitState",
    "LLMRequestEnvelope",
]


class EnvelopeStatus(StrEnum):
    """Terminal status of an LLM request."""

    OK = "ok"
    CACHED = "cached"
    ERROR = "error"
    DEGRADED = "degraded"
    DENIED = "denied"


class CostSource(StrEnum):
    """Provenance of cost estimation."""

    ESTIMATED_LOCAL_SNAPSHOT = "estimated_local_snapshot"
    ESTIMATED_PROVIDER_API = "estimated_provider_api"
    ACTUAL_PROVIDER_INVOICE = "actual_provider_invoice"
    CACHED_ZERO = "cached_zero"
    DEGRADED_UNKNOWN = "degraded_unknown"


class CircuitState(StrEnum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class LLMRequestEnvelope:
    """
    The envelope contract for LLM request lifecycle.

    This dataclass models all six semantic blocks:
    - identity/context
    - model selection
    - economics
    - reliability
    - governance
    - cache/eval
    """

    # Schema versioning
    schema_version: str

    # Identity and context
    request_id: str
    tenant_id: str
    route: str
    trace_id: str | None = None
    span_id: str | None = None
    caller_id: str | None = None
    use_case: str | None = None
    session_id: str | None = None
    task_id: str | None = None

    # Model selection
    provider_requested: str | None = None
    model_requested: str | None = None
    provider_selected: str | None = None
    model_selected: str | None = None
    model_tier: str | None = None
    routing_decision: str | None = None
    routing_reason: str | None = None

    # Economics
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_total: int | None = None
    estimated_cost_usd: float | None = None
    cost_source: CostSource = CostSource.DEGRADED_UNKNOWN

    # Reliability
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    status: EnvelopeStatus = EnvelopeStatus.OK
    error_type: str | None = None
    retry_count: int | None = None
    fallback_triggered: bool | None = None
    fallback_reason: str | None = None
    circuit_state: CircuitState | None = None
    streaming: bool | None = None
    finish_reason: str | None = None

    # Governance
    policy_input_class: str | None = None
    policy_decision: str | None = None
    policy_mode: str | None = None
    redaction_applied: bool | None = None
    pii_detected: bool | None = None

    # Cache and evaluation
    cache_eligible: bool | None = None
    cache_strategy: str | None = None
    cache_hit: bool | None = None
    cache_key_fingerprint: str | None = None
    cache_key_algorithm: str | None = None
    cache_lookup_confidence: float | None = None
    eval_hooks: tuple[str, ...] = field(default_factory=tuple)
    audit_tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert envelope to dictionary for serialization."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (tuple, dict)) and not value:
                continue
            elif value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMRequestEnvelope":
        """Reconstruct an envelope from a dictionary (e.g. parsed JSONL).

        String values for enum fields are coerced to the corresponding
        StrEnum member.  Unknown keys are silently ignored so that
        forward-compatible payloads (with fields added in newer schema
        versions) can still be loaded by older library versions.

        Args:
            data: Dictionary with envelope fields.

        Returns:
            LLMRequestEnvelope instance.
        """
        _enum_fields: dict[str, type[StrEnum]] = {
            "status": EnvelopeStatus,
            "cost_source": CostSource,
            "circuit_state": CircuitState,
        }
        known_fields = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        for key, value in data.items():
            if key not in known_fields:
                continue
            if key in _enum_fields and isinstance(value, str):
                kwargs[key] = _enum_fields[key](value)
            elif key == "eval_hooks" and isinstance(value, (list, tuple)):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value

        return cls(**kwargs)
