"""Regression test: _record_otel_metrics prefers metadata["model_tier"]
over metadata["routing_decision"] for the model_tier metric dimension."""

from unittest.mock import patch

from turnpike.gateway.telemetry import _record_otel_metrics


class TestModelTierDimension:
    """Verify model_tier metric dimension resolution order."""

    @patch("turnpike.gateway.telemetry._request_counter")
    @patch("turnpike.gateway.telemetry._cost_counter")
    @patch("turnpike.gateway.telemetry._operation_duration_histogram")
    def test_prefers_model_tier_over_routing_decision(self, mock_dur, mock_cost, mock_req):
        _ = mock_dur
        _record_otel_metrics(
            route="/summarize",
            provider_system="openai",
            model="gpt-4o-mini",
            latency_ms=100.0,
            status="success",
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            error_type=None,
            metadata={"model_tier": "cheap", "routing_decision": "tier_based"},
        )
        cost_attrs = mock_cost.add.call_args[1]["attributes"]
        req_attrs = mock_req.add.call_args[1]["attributes"]
        assert cost_attrs["turnpike.model_tier"] == "cheap"
        assert req_attrs["turnpike.model_tier"] == "cheap"

    @patch("turnpike.gateway.telemetry._request_counter")
    @patch("turnpike.gateway.telemetry._cost_counter")
    @patch("turnpike.gateway.telemetry._operation_duration_histogram")
    def test_falls_back_to_routing_decision(self, mock_dur, mock_cost, mock_req):
        _ = mock_dur
        _record_otel_metrics(
            route="/classify",
            provider_system="openai",
            model="gpt-4o-mini",
            latency_ms=50.0,
            status="success",
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            error_type=None,
            metadata={"routing_decision": "tier_based"},
        )
        cost_attrs = mock_cost.add.call_args[1]["attributes"]
        assert cost_attrs["turnpike.model_tier"] == "tier_based"

    @patch("turnpike.gateway.telemetry._request_counter")
    @patch("turnpike.gateway.telemetry._cost_counter")
    @patch("turnpike.gateway.telemetry._operation_duration_histogram")
    def test_defaults_to_unknown(self, mock_dur, mock_cost, mock_req):
        _ = mock_dur
        _record_otel_metrics(
            route="/extract",
            provider_system="openai",
            model="gpt-4o-mini",
            latency_ms=50.0,
            status="success",
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            error_type=None,
            metadata=None,
        )
        cost_attrs = mock_cost.add.call_args[1]["attributes"]
        assert cost_attrs["turnpike.model_tier"] == "unknown"
