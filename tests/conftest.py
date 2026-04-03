"""Shared test fixtures and configuration."""

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch fixture."""
    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session", autouse=True)
def set_app_api_key(monkeypatch_session):
    """Set APP_API_KEY for all tests in the session."""
    monkeypatch_session.setenv("APP_API_KEY", "test-key-007")


@pytest.fixture(scope="session", autouse=True)
def register_test_route_policies():
    """Register route policies needed by gateway and reference-app tests."""
    from turnpike.gateway.policies import RoutePolicy, register_route_policy

    _policy = RoutePolicy(
        max_output_tokens=500,
        retry_attempts=2,
        cache_enabled=False,
        model_for_tier={"cheap": "gpt-4o-mini", "expensive": "gpt-4o"},
        provider_name="openai",
    )
    register_route_policy("/answer-routed", _policy)
    register_route_policy("/conversation-turn", _policy)


@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    """Reset rate limit windows before each test for isolation."""
    from app.middleware.rate_limit import reset_rate_limit_windows

    reset_rate_limit_windows()
    yield
    reset_rate_limit_windows()
