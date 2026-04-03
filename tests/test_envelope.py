"""
Tests for core.envelope — the portable envelope contract.
"""

from turnpike.envelope import CircuitState, CostSource, EnvelopeStatus, LLMRequestEnvelope


def test_envelope_imports_without_framework():
    """Verify turnpike.envelope does not import fastapi or starlette as dependencies."""
    # This test verifies that importing turnpike.envelope doesn't require framework packages
    # by checking the module's __dict__ for framework references
    import turnpike.envelope

    module_dict = dir(turnpike.envelope)
    # Verify no fastapi or starlette types are exposed in the module
    assert not any("fastapi" in str(item).lower() for item in module_dict)
    assert not any("starlette" in str(item).lower() for item in module_dict)


def test_cost_source_enum_values_exist():
    """Verify all CostSource enum values are defined."""
    assert CostSource.ESTIMATED_LOCAL_SNAPSHOT == "estimated_local_snapshot"
    assert CostSource.ESTIMATED_PROVIDER_API == "estimated_provider_api"
    assert CostSource.ACTUAL_PROVIDER_INVOICE == "actual_provider_invoice"
    assert CostSource.CACHED_ZERO == "cached_zero"
    assert CostSource.DEGRADED_UNKNOWN == "degraded_unknown"


def test_circuit_state_enum_values_exist():
    """Verify all CircuitState enum values are defined."""
    assert CircuitState.CLOSED == "closed"
    assert CircuitState.OPEN == "open"
    assert CircuitState.HALF_OPEN == "half_open"


def test_envelope_status_enum_values_exist():
    """Verify all EnvelopeStatus enum values are defined."""
    assert EnvelopeStatus.OK == "ok"
    assert EnvelopeStatus.CACHED == "cached"
    assert EnvelopeStatus.ERROR == "error"
    assert EnvelopeStatus.DEGRADED == "degraded"
    assert EnvelopeStatus.DENIED == "denied"


def test_minimal_envelope_can_be_created():
    """Verify envelope can be instantiated with minimal required fields."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
    )
    assert envelope.schema_version == "0.1.0"
    assert envelope.request_id == "req-123"
    assert envelope.tenant_id == "tenant-abc"
    assert envelope.route == "/answer-routed"


def test_envelope_has_cache_key_algorithm_field():
    """Verify cache_key_algorithm field is present."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
        cache_key_algorithm="sha256",
    )
    assert envelope.cache_key_algorithm == "sha256"


def test_envelope_has_schema_version_field():
    """Verify schema_version field is present and required."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
    )
    assert envelope.schema_version == "0.1.0"


def test_envelope_has_cost_source_field():
    """Verify cost_source field is present with default."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
    )
    assert envelope.cost_source == CostSource.DEGRADED_UNKNOWN


def test_envelope_cost_source_can_be_set():
    """Verify cost_source can be explicitly set."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
        cost_source=CostSource.ESTIMATED_LOCAL_SNAPSHOT,
    )
    assert envelope.cost_source == CostSource.ESTIMATED_LOCAL_SNAPSHOT


def test_envelope_to_dict_serialization():
    """Verify to_dict() converts envelope to dictionary."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
        cost_source=CostSource.CACHED_ZERO,
        status=EnvelopeStatus.CACHED,
        circuit_state=CircuitState.CLOSED,
    )
    result = envelope.to_dict()
    assert result["schema_version"] == "0.1.0"
    assert result["request_id"] == "req-123"
    assert result["cost_source"] == "cached_zero"
    assert result["status"] == "cached"
    assert result["circuit_state"] == "closed"


def test_envelope_to_dict_omits_none_values():
    """Verify to_dict() omits None values."""
    envelope = LLMRequestEnvelope(
        schema_version="0.1.0",
        request_id="req-123",
        tenant_id="tenant-abc",
        route="/answer-routed",
    )
    result = envelope.to_dict()
    assert "trace_id" not in result
    assert "span_id" not in result
    assert "tokens_in" not in result
