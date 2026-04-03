"""Integration tests for context propagation through call_llm."""

from unittest.mock import AsyncMock, patch

import pytest

from turnpike.context import LLMRequestContext
from turnpike.gateway.client import call_llm
from turnpike.gateway.provider import ProviderResponse


@pytest.fixture
def mock_provider_response():
    """Mock provider response for testing."""
    return ProviderResponse(
        text="Test response",
        tokens_in=10,
        tokens_out=20,
    )


class TestContextPropagation:
    """Tests for context propagation through call_llm."""

    async def test_call_llm_with_metadata_only_backward_compat(
        self, mock_provider_response, tmp_path, monkeypatch
    ):
        """Existing metadata-only callers should continue to work."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = (
                mock_provider_response.text,
                mock_provider_response.tokens_in,
                mock_provider_response.tokens_out,
                0,
                "stop",
            )

            metadata = {
                "tenant_id": "metadata-tenant",
                "use_case": "metadata-use-case",
            }

            result = await call_llm(
                prompt="test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
                metadata=metadata,
            )

            assert result.text == "Test response"
            assert result.tokens_in == 10
            assert result.tokens_out == 20

            # Verify JSONL contains metadata-derived context
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "metadata-tenant"
                assert event["use_case"] == "metadata-use-case"

    async def test_call_llm_with_context_takes_precedence(
        self, mock_provider_response, tmp_path, monkeypatch
    ):
        """Context parameter should take precedence over metadata."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = (
                mock_provider_response.text,
                mock_provider_response.tokens_in,
                mock_provider_response.tokens_out,
                0,
                "stop",
            )

            metadata = {
                "tenant_id": "metadata-tenant",
                "use_case": "metadata-use-case",
            }

            context = LLMRequestContext(
                tenant_id="context-tenant",
                use_case="context-use-case",
                caller_id="api-gateway",
            )

            result = await call_llm(
                prompt="test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
                metadata=metadata,
                context=context,
            )

            assert result.text == "Test response"

            # Verify JSONL contains context values, not metadata
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "context-tenant"
                assert event["use_case"] == "context-use-case"
                assert event["caller_id"] == "api-gateway"

    async def test_call_llm_without_context_uses_default_tenant(
        self, mock_provider_response, tmp_path, monkeypatch
    ):
        """When no context or metadata provided, should use default tenant."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = (
                mock_provider_response.text,
                mock_provider_response.tokens_in,
                mock_provider_response.tokens_out,
                0,
                "stop",
            )

            result = await call_llm(
                prompt="test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
            )

            assert result.text == "Test response"

            # Verify JSONL contains default tenant
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "default"

    async def test_call_llm_with_audit_tags_in_envelope(
        self, mock_provider_response, tmp_path, monkeypatch
    ):
        """Context audit tags should appear in envelope."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = (
                mock_provider_response.text,
                mock_provider_response.tokens_in,
                mock_provider_response.tokens_out,
                0,
                "stop",
            )

            context = LLMRequestContext(
                tenant_id="acme-corp",
                feature_id="smart-routing-v3",
                experiment_id="exp-2024-q1",
                budget_namespace="support-team",
            )

            result = await call_llm(
                prompt="test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
                context=context,
            )

            assert result.text == "Test response"

            # Verify JSONL contains audit tags
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "acme-corp"
                assert event["audit_tags"] == {
                    "feature_id": "smart-routing-v3",
                    "experiment_id": "exp-2024-q1",
                    "budget_namespace": "support-team",
                }

    async def test_call_llm_error_path_preserves_context(self, tmp_path, monkeypatch):
        """Context should be preserved in envelope even on error."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.side_effect = RuntimeError("Provider error")

            context = LLMRequestContext(
                tenant_id="acme-corp",
                caller_id="api-gateway",
                use_case="customer-support",
            )

            with pytest.raises(RuntimeError, match="Provider error"):
                await call_llm(
                    prompt="test prompt",
                    model_tier="cheap",
                    route_name="/answer-routed",
                    context=context,
                )

            # Verify JSONL error event contains context
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "acme-corp"
                assert event["caller_id"] == "api-gateway"
                assert event["use_case"] == "customer-support"
                assert event["status"] == "error"

    async def test_call_llm_empty_context_uses_defaults(
        self, mock_provider_response, tmp_path, monkeypatch
    ):
        """Empty context should result in default values."""
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr("turnpike.gateway.telemetry.TELEMETRY_PATH", test_telemetry)

        with patch(
            "turnpike.gateway.client._call_provider",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = (
                mock_provider_response.text,
                mock_provider_response.tokens_in,
                mock_provider_response.tokens_out,
                0,
                "stop",
            )

            context = LLMRequestContext()  # All None

            result = await call_llm(
                prompt="test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
                context=context,
            )

            assert result.text == "Test response"

            # Verify JSONL uses defaults
            import json

            with open(test_telemetry) as f:
                event = json.loads(f.read())
                assert event["tenant_id"] == "default"
                assert event.get("caller_id") is None
                assert event.get("use_case") is None
                assert event.get("audit_tags", {}) == {}
