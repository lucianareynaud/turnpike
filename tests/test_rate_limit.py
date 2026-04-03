"""Tests for rate limiting middleware."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def rate_limit_env(monkeypatch):
    """Set RATE_LIMIT_RPM to 3 for all rate limit tests."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "3")


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def test_under_limit_passes(self):
        """Two requests under limit of 3 should both pass."""
        response1 = client.post(
            "/classify-complexity",
            json={"message": "first"},
            headers={"X-API-Key": "test-key-007"},
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/classify-complexity",
            json={"message": "second"},
            headers={"X-API-Key": "test-key-007"},
        )
        assert response2.status_code == 200

    def test_at_limit_passes(self):
        """Exactly 3 requests should all pass."""
        for i in range(3):
            response = client.post(
                "/classify-complexity",
                json={"message": f"request {i}"},
                headers={"X-API-Key": "test-key-007"},
            )
            assert response.status_code == 200

    def test_over_limit_rejected(self):
        """Fourth request should be rejected with 429."""
        # autouse reset_rate_limit_state in conftest ensures clean state.
        for i in range(3):
            response = client.post(
                "/classify-complexity",
                json={"message": f"request {i}"},
                headers={"X-API-Key": "test-key-007"},
            )
            assert response.status_code == 200

        response = client.post(
            "/classify-complexity",
            json={"message": "request 4"},
            headers={"X-API-Key": "test-key-007"},
        )
        assert response.status_code == 429
        assert response.json() == {"detail": "Rate limit exceeded"}
        assert response.headers.get("Retry-After") == "60"

    def test_different_keys_independent(self):
        """Different API keys should have independent rate limit windows.

        Since APIKeyMiddleware only accepts APP_API_KEY, both callers here
        use the same valid key. The caller_id used for rate-limit bucketing
        is set by resolve_caller() in auth.py. With a single shared APP_API_KEY,
        both callers share the same bucket. Multi-key isolation requires
        multi-tenant auth support (future spec).
        """
        pytest.skip("Multi-key rate limiting requires multi-tenant auth support")
