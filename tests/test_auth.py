"""Tests for authentication middleware and pure functions."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import authenticate, resolve_caller


# Pure function tests — no TestClient, no middleware instantiation
class TestAuthenticatePureFunction:
    """Tests for authenticate() pure function."""

    def test_authenticate_correct_key(self):
        assert authenticate("secret", "secret") is True

    def test_authenticate_wrong_key(self):
        assert authenticate("wrong", "secret") is False

    def test_authenticate_empty_provided(self):
        assert authenticate("", "secret") is False


class TestResolveCallerPureFunction:
    """Tests for resolve_caller() pure function."""

    def test_resolve_caller_no_header(self):
        caller_id, tenant_id = resolve_caller({})
        assert caller_id == "default"
        assert tenant_id == "default"

    def test_resolve_caller_uses_api_key(self):
        caller_id, tenant_id = resolve_caller({"x-api-key": "abc"})
        assert caller_id == "abc"
        assert tenant_id == "default"

    def test_resolve_caller_tenant_always_default(self):
        # Verify tenant_id is always "default" regardless of headers
        _, tenant_id = resolve_caller({"x-api-key": "key1", "x-tenant-id": "custom"})
        assert tenant_id == "default"


# Middleware integration tests — use TestClient
class TestAPIKeyMiddlewareIntegration:
    """Integration tests for APIKeyMiddleware."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create test client with APP_API_KEY set."""
        monkeypatch.setenv("APP_API_KEY", "test-key-007")
        return TestClient(app, raise_server_exceptions=False)

    def test_valid_key_passes(self, client):
        response = client.post(
            "/classify-complexity",
            json={"message": "hi"},
            headers={"X-API-Key": "test-key-007"},
        )
        assert response.status_code == 200

    def test_missing_key_rejected(self, client):
        response = client.post("/classify-complexity", json={"message": "hi"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    def test_wrong_key_rejected(self, client):
        response = client.post(
            "/classify-complexity",
            json={"message": "hi"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    def test_healthz_exempt(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_readyz_exempt(self, client):
        response = client.get("/readyz")
        assert response.status_code in [200, 503]
