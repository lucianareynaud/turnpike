"""Tests for gateway components.

MOCK PATTERN
Tests that call ``call_llm()`` patch ``gateway.client.get_provider``
to return a mock provider conforming to ProviderBase.

ERROR CLASSIFICATION TESTS
``categorize_error()`` and ``is_retryable()`` are methods on provider
implementations. Tests construct real SDK exceptions and call directly.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest

from turnpike.gateway import cost_model, policies
from turnpike.gateway.client import GatewayResult, call_llm
from turnpike.gateway.provider import AnthropicProvider, OpenAIProvider, ProviderResponse


def _make_response(status_code: int) -> httpx.Response:
    """Build a minimal fake httpx.Response for OpenAI exceptions."""
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/test"),
    )


def _make_request() -> httpx.Request:
    """Build a minimal fake httpx.Request for connection errors."""
    return httpx.Request("POST", "https://api.openai.com/v1/test")


def _build_mock_provider(
    text: str = "Test response",
    tokens_in: int = 100,
    tokens_out: int = 50,
    provider_name: str = "openai",
) -> Mock:
    """Build a mock provider conforming to ProviderBase."""
    mock = Mock()
    mock.provider_name = provider_name
    mock.complete = AsyncMock(
        return_value=ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    )
    mock.is_retryable = Mock(return_value=False)
    mock.categorize_error = Mock(return_value="unknown")
    return mock


class TestCostModel:
    """Tests for cost_model module."""

    def test_estimate_cost_gpt4o_mini(self):
        cost = cost_model.estimate_cost("gpt-4o-mini", 1_000_000, 500_000)
        assert cost == pytest.approx(0.45)

    def test_estimate_cost_gpt4o(self):
        cost = cost_model.estimate_cost("gpt-4o", 1_000_000, 500_000)
        assert cost == pytest.approx(7.50)

    def test_estimate_cost_claude_sonnet(self):
        cost = cost_model.estimate_cost("claude-sonnet-4-20250514", 1_000_000, 500_000)
        assert cost == pytest.approx(10.50)

    def test_estimate_cost_claude_haiku(self):
        cost = cost_model.estimate_cost("claude-haiku-3-5-20241022", 1_000_000, 500_000)
        assert cost == pytest.approx(2.80)

    def test_estimate_cost_small_values(self):
        cost = cost_model.estimate_cost("gpt-4o-mini", 1000, 500)
        assert cost == pytest.approx(0.00045)

    def test_estimate_cost_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            cost_model.estimate_cost("unknown-model", 1000, 500)

    def test_estimate_cost_negative_tokens(self):
        with pytest.raises(ValueError, match="non-negative"):
            cost_model.estimate_cost("gpt-4o-mini", -100, 500)
        with pytest.raises(ValueError, match="non-negative"):
            cost_model.estimate_cost("gpt-4o-mini", 1000, -50)

    def test_get_pricing(self):
        pricing = cost_model.get_pricing()
        assert "gpt-4o-mini" in pricing
        assert "gpt-4o" in pricing
        assert "claude-sonnet-4-20250514" in pricing
        assert "claude-haiku-3-5-20241022" in pricing
        assert pricing["gpt-4o-mini"]["input_per_1m"] == 0.15
        assert pricing["gpt-4o"]["output_per_1m"] == 10.00
        assert pricing["claude-sonnet-4-20250514"]["input_per_1m"] == 3.00
        assert pricing["claude-haiku-3-5-20241022"]["output_per_1m"] == 4.00


class TestPolicies:
    """Tests for policies module."""

    def setup_method(self):
        """Register test routes before each test."""
        policies.register_route_policy(
            "/test-route",
            policies.RoutePolicy(
                max_output_tokens=500,
                retry_attempts=2,
                cache_enabled=False,
                model_for_tier={"cheap": "gpt-4o-mini", "expensive": "gpt-4o"},
                provider_name="openai",
            ),
        )

    def teardown_method(self):
        if "/test-route" in policies._ROUTE_POLICIES:
            del policies._ROUTE_POLICIES["/test-route"]

    def test_get_registered_route_policy(self):
        policy = policies.get_route_policy("/test-route")
        assert policy.max_output_tokens == 500
        assert policy.retry_attempts == 2
        assert policy.cache_enabled is False
        assert "cheap" in policy.model_for_tier
        assert "expensive" in policy.model_for_tier
        assert policy.provider_name == "openai"

    def test_get_route_policy_unknown_route(self):
        with pytest.raises(ValueError, match="No gateway policy"):
            policies.get_route_policy("/unknown-route")

    def test_get_route_policy_empty_registry(self):
        saved = dict(policies._ROUTE_POLICIES)
        try:
            policies.clear_route_policies()
            with pytest.raises(ValueError, match="register_route_policy"):
                policies.get_route_policy("/anything")
        finally:
            policies._ROUTE_POLICIES.update(saved)

    def test_get_model_for_tier_cheap(self):
        model = policies.get_model_for_tier("/test-route", "cheap")
        assert model == "gpt-4o-mini"

    def test_get_model_for_tier_expensive(self):
        model = policies.get_model_for_tier("/test-route", "expensive")
        assert model == "gpt-4o"

    def test_get_model_for_tier_unknown_tier(self):
        with pytest.raises(ValueError, match="not configured"):
            policies.get_model_for_tier(
                "/test-route",
                "invalid",  # type: ignore[arg-type]
            )

    def test_route_policy_default_provider(self):
        policy = policies.RoutePolicy(
            max_output_tokens=100,
            retry_attempts=1,
            cache_enabled=False,
            model_for_tier={
                "cheap": "gpt-4o-mini",
                "expensive": "gpt-4o",
            },
        )
        assert policy.provider_name == "openai"


class TestGatewayClient:
    """Tests for gateway client module."""

    def test_gateway_result_structure(self):
        result = GatewayResult(
            text="test response",
            selected_model="gpt-4o-mini",
            request_id="test-id",
            tokens_in=100,
            tokens_out=50,
            estimated_cost_usd=0.001,
            cache_hit=False,
        )
        assert result.text == "test response"
        assert result.selected_model == "gpt-4o-mini"
        assert result.request_id == "test-id"
        assert result.tokens_in == 100
        assert result.tokens_out == 50
        assert result.estimated_cost_usd == 0.001
        assert result.cache_hit is False

    @patch("turnpike.gateway.client.get_provider")
    async def test_call_llm_success(self, mock_get_provider, tmp_path, monkeypatch):
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr(
            "turnpike.gateway.telemetry.TELEMETRY_PATH",
            test_telemetry,
        )

        mock_provider = _build_mock_provider(text="Test response", tokens_in=100, tokens_out=50)
        mock_get_provider.return_value = mock_provider

        result = await call_llm(
            prompt="Test prompt",
            model_tier="cheap",
            route_name="/answer-routed",
            metadata={"routing_decision": "cheap"},
        )

        assert isinstance(result, GatewayResult)
        assert result.text == "Test response"
        assert result.selected_model == "gpt-4o-mini"
        assert result.tokens_in == 100
        assert result.tokens_out == 50
        assert result.estimated_cost_usd > 0
        assert result.cache_hit is False
        assert len(result.request_id) > 0

        mock_provider.complete.assert_awaited_once_with(
            [{"role": "user", "content": "Test prompt"}], "gpt-4o-mini", 500
        )

        assert test_telemetry.exists()

        with test_telemetry.open() as fh:
            event = json.loads(fh.readline())

        assert event["status"] == "success"
        assert event["route"] == "/answer-routed"
        assert event["model"] == "gpt-4o-mini"
        assert event["tokens_in"] == 100
        assert event["tokens_out"] == 50
        assert event["routing_decision"] == "cheap"
        assert event["selected_model"] == "gpt-4o-mini"

    @patch("turnpike.gateway.client.get_provider")
    async def test_call_llm_error(self, mock_get_provider, tmp_path, monkeypatch):
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr(
            "turnpike.gateway.telemetry.TELEMETRY_PATH",
            test_telemetry,
        )

        exc = openai.BadRequestError(
            "invalid request",
            response=_make_response(400),
            body=None,
        )

        mock_provider = _build_mock_provider()
        mock_provider.complete = AsyncMock(side_effect=exc)
        mock_provider.is_retryable = Mock(return_value=False)
        mock_provider.categorize_error = Mock(return_value="invalid_request")
        mock_get_provider.return_value = mock_provider

        with pytest.raises(openai.BadRequestError):
            await call_llm(
                prompt="Test prompt",
                model_tier="cheap",
                route_name="/answer-routed",
            )

        assert test_telemetry.exists()

        with test_telemetry.open() as fh:
            event = json.loads(fh.readline())

        assert event["status"] == "error"
        assert event["error_type"] == "invalid_request"
        assert event["tokens_in"] == 0
        assert event["tokens_out"] == 0


class TestOpenAIProviderErrorHandling:
    """Tests for OpenAIProvider error classification."""

    def setup_method(self):
        self.provider = OpenAIProvider()

    def test_categorize_error_rate_limit(self):
        error = openai.RateLimitError(
            "rate limit exceeded",
            response=_make_response(429),
            body=None,
        )
        assert self.provider.categorize_error(error) == "rate_limit"

    def test_categorize_error_invalid_request(self):
        error = openai.BadRequestError(
            "invalid request",
            response=_make_response(400),
            body=None,
        )
        result = self.provider.categorize_error(error)
        assert result == "invalid_request"

    def test_categorize_error_auth(self):
        error = openai.AuthenticationError(
            "incorrect API key",
            response=_make_response(401),
            body=None,
        )
        assert self.provider.categorize_error(error) == "auth_error"

    def test_categorize_error_permission_denied(self):
        error = openai.PermissionDeniedError(
            "forbidden",
            response=_make_response(403),
            body=None,
        )
        assert self.provider.categorize_error(error) == "auth_error"

    def test_categorize_error_timeout(self):
        error = openai.APITimeoutError(request=_make_request())
        assert self.provider.categorize_error(error) == "timeout"

    def test_categorize_error_transient_connection(self):
        error = openai.APIConnectionError(request=_make_request())
        result = self.provider.categorize_error(error)
        assert result == "transient_error"

    def test_categorize_error_transient_server(self):
        error = openai.InternalServerError(
            "server error",
            response=_make_response(503),
            body=None,
        )
        result = self.provider.categorize_error(error)
        assert result == "transient_error"

    def test_categorize_error_unknown(self):
        error = Exception("something unexpected")
        assert self.provider.categorize_error(error) == "unknown"

    def test_is_retryable_rate_limit(self):
        error = openai.RateLimitError(
            "rate limit",
            response=_make_response(429),
            body=None,
        )
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_timeout(self):
        error = openai.APITimeoutError(request=_make_request())
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_transient_server_error(self):
        error = openai.InternalServerError(
            "server error",
            response=_make_response(503),
            body=None,
        )
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_connection_error(self):
        error = openai.APIConnectionError(request=_make_request())
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_invalid_request(self):
        error = openai.BadRequestError(
            "invalid request",
            response=_make_response(400),
            body=None,
        )
        assert self.provider.is_retryable(error) is False

    def test_is_retryable_auth_error(self):
        error = openai.AuthenticationError(
            "bad key",
            response=_make_response(401),
            body=None,
        )
        assert self.provider.is_retryable(error) is False

    def test_is_retryable_plain_exception(self):
        error = Exception("something unexpected")
        assert self.provider.is_retryable(error) is False


def _make_anthropic_response(status_code: int) -> httpx.Response:
    """Build a minimal fake httpx.Response for Anthropic exceptions."""
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def _make_anthropic_request() -> httpx.Request:
    """Build a minimal fake httpx.Request for Anthropic connection errors."""
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class TestAnthropicProviderErrorHandling:
    """Tests for AnthropicProvider error classification."""

    anthropic = pytest.importorskip("anthropic")

    def setup_method(self):
        self.provider = AnthropicProvider()

    def test_categorize_error_rate_limit(self):
        error = self.anthropic.RateLimitError(
            "rate limit exceeded",
            response=_make_anthropic_response(429),
            body=None,
        )
        assert self.provider.categorize_error(error) == "rate_limit"

    def test_categorize_error_auth(self):
        error = self.anthropic.AuthenticationError(
            "invalid api key",
            response=_make_anthropic_response(401),
            body=None,
        )
        assert self.provider.categorize_error(error) == "auth_error"

    def test_categorize_error_timeout(self):
        error = self.anthropic.APITimeoutError(request=_make_anthropic_request())
        assert self.provider.categorize_error(error) == "timeout"

    def test_categorize_error_transient(self):
        error = self.anthropic.InternalServerError(
            "server error",
            response=_make_anthropic_response(500),
            body=None,
        )
        assert self.provider.categorize_error(error) == "transient_error"

    def test_categorize_error_invalid_request(self):
        error = self.anthropic.BadRequestError(
            "bad request",
            response=_make_anthropic_response(400),
            body=None,
        )
        assert self.provider.categorize_error(error) == "invalid_request"

    def test_categorize_error_unknown(self):
        error = Exception("something unexpected")
        assert self.provider.categorize_error(error) == "unknown"

    def test_is_retryable_rate_limit(self):
        error = self.anthropic.RateLimitError(
            "rate limit",
            response=_make_anthropic_response(429),
            body=None,
        )
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_connection_error(self):
        error = self.anthropic.APIConnectionError(request=_make_anthropic_request())
        assert self.provider.is_retryable(error) is True

    def test_is_retryable_invalid_request(self):
        error = self.anthropic.BadRequestError(
            "bad request",
            response=_make_anthropic_response(400),
            body=None,
        )
        assert self.provider.is_retryable(error) is False

    def test_is_retryable_plain_exception(self):
        error = Exception("something unexpected")
        assert self.provider.is_retryable(error) is False


class TestProviderRegistry:
    """Tests for provider registration and resolution."""

    def test_openai_auto_registered(self):
        from turnpike.gateway.provider import available_providers

        assert "openai" in available_providers()

    def test_get_provider_unknown(self):
        from turnpike.gateway.provider import get_provider

        with pytest.raises(ValueError, match="not registered"):
            get_provider("nonexistent-provider")

    def test_provider_name_property(self):
        provider = OpenAIProvider()
        assert provider.provider_name == "openai"

    def test_provider_base_defaults(self):
        """ProviderBase defaults: safe values for unknown errors."""
        provider = OpenAIProvider()
        plain_error = Exception("unknown")
        assert provider.categorize_error(plain_error) == "unknown"
        assert provider.is_retryable(plain_error) is False


class TestTelemetryEmission:
    """Tests for telemetry emission."""

    def test_telemetry_event_shape(self, tmp_path, monkeypatch):
        from turnpike.gateway import telemetry

        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr(
            "turnpike.gateway.telemetry.TELEMETRY_PATH",
            test_telemetry,
        )

        telemetry.emit(
            request_id="test-id",
            route="/answer-routed",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=123.45,
            status="success",
            tokens_in=100,
            tokens_out=50,
            estimated_cost_usd=0.001,
            cache_hit=False,
            schema_valid=True,
            error_type=None,
            metadata={
                "routing_decision": "cheap",
                "selected_model": "gpt-4o-mini",
            },
        )

        assert test_telemetry.exists()

        with test_telemetry.open() as fh:
            event = json.loads(fh.readline())

        assert "timestamp" in event
        assert event["request_id"] == "test-id"
        assert event["route"] == "/answer-routed"
        assert event["provider"] == "openai"
        assert event["model"] == "gpt-4o-mini"
        assert event["latency_ms"] == 123.45
        assert event["status"] == "success"
        assert event["tokens_in"] == 100
        assert event["tokens_out"] == 50
        assert event["estimated_cost_usd"] == 0.001
        assert event["cache_hit"] is False
        assert event["schema_valid"] is True
        assert event["error_type"] is None
        assert event["routing_decision"] == "cheap"
        assert event["selected_model"] == "gpt-4o-mini"

    @patch("turnpike.gateway.client.get_provider")
    async def test_envelope_in_telemetry_path(self, mock_get_provider, tmp_path, monkeypatch):
        test_telemetry = tmp_path / "telemetry.jsonl"
        monkeypatch.setattr(
            "turnpike.gateway.telemetry.TELEMETRY_PATH",
            test_telemetry,
        )

        mock_provider = _build_mock_provider(
            text="Test response",
            tokens_in=100,
            tokens_out=50,
        )
        mock_get_provider.return_value = mock_provider

        await call_llm(
            prompt="Test prompt",
            model_tier="cheap",
            route_name="/answer-routed",
        )

        assert test_telemetry.exists()

        with test_telemetry.open() as fh:
            event = json.loads(fh.readline())

        assert event["schema_version"] == "0.1.0"
        assert event["cost_source"] == "estimated_local_snapshot"
        assert event["tenant_id"] == "default"
        assert event["tokens_total"] == 150
