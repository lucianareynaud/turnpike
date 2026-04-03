"""Tests for turnpike.context — structured request context."""

import pytest

from turnpike.context import LLMRequestContext


class TestLLMRequestContext:
    """Tests for LLMRequestContext dataclass."""

    def test_empty_context_creation(self):
        """Empty context should have all None fields."""
        ctx = LLMRequestContext()

        assert ctx.tenant_id is None
        assert ctx.caller_id is None
        assert ctx.use_case is None
        assert ctx.feature_id is None
        assert ctx.experiment_id is None
        assert ctx.budget_namespace is None

    def test_partial_context_creation(self):
        """Context with some fields set should preserve values."""
        ctx = LLMRequestContext(
            tenant_id="acme-corp",
            use_case="customer-support",
        )

        assert ctx.tenant_id == "acme-corp"
        assert ctx.use_case == "customer-support"
        assert ctx.caller_id is None
        assert ctx.feature_id is None

    def test_full_context_creation(self):
        """Context with all fields set should preserve all values."""
        ctx = LLMRequestContext(
            tenant_id="acme-corp",
            caller_id="support-bot-v2",
            use_case="customer-support",
            feature_id="smart-routing-v3",
            experiment_id="exp-2024-q1-routing",
            budget_namespace="support-team",
        )

        assert ctx.tenant_id == "acme-corp"
        assert ctx.caller_id == "support-bot-v2"
        assert ctx.use_case == "customer-support"
        assert ctx.feature_id == "smart-routing-v3"
        assert ctx.experiment_id == "exp-2024-q1-routing"
        assert ctx.budget_namespace == "support-team"

    def test_context_is_frozen(self):
        """Context should be immutable after creation."""
        ctx = LLMRequestContext(tenant_id="acme-corp")

        with pytest.raises((AttributeError, TypeError)):
            ctx.tenant_id = "other-corp"  # type: ignore[misc]

    def test_from_metadata_with_none(self):
        """from_metadata with None should return empty context."""
        ctx = LLMRequestContext.from_metadata(None)

        assert ctx.tenant_id is None
        assert ctx.caller_id is None
        assert ctx.use_case is None

    def test_from_metadata_with_empty_dict(self):
        """from_metadata with empty dict should return empty context."""
        ctx = LLMRequestContext.from_metadata({})

        assert ctx.tenant_id is None
        assert ctx.caller_id is None

    def test_from_metadata_extracts_known_fields(self):
        """from_metadata should extract all known attribution fields."""
        metadata = {
            "tenant_id": "acme-corp",
            "caller_id": "api-gateway",
            "use_case": "code-review",
            "feature_id": "ai-suggestions",
            "experiment_id": "exp-123",
            "budget_namespace": "eng-team",
        }

        ctx = LLMRequestContext.from_metadata(metadata)

        assert ctx.tenant_id == "acme-corp"
        assert ctx.caller_id == "api-gateway"
        assert ctx.use_case == "code-review"
        assert ctx.feature_id == "ai-suggestions"
        assert ctx.experiment_id == "exp-123"
        assert ctx.budget_namespace == "eng-team"

    def test_from_metadata_ignores_unknown_fields(self):
        """from_metadata should ignore fields not in context schema."""
        metadata = {
            "tenant_id": "acme-corp",
            "unknown_field": "should-be-ignored",
            "routing_decision": "cheap",  # route-specific, not context
        }

        ctx = LLMRequestContext.from_metadata(metadata)

        assert ctx.tenant_id == "acme-corp"
        assert not hasattr(ctx, "unknown_field")
        assert not hasattr(ctx, "routing_decision")

    def test_to_audit_tags_empty_context(self):
        """to_audit_tags on empty context should return empty dict."""
        ctx = LLMRequestContext()

        tags = ctx.to_audit_tags()

        assert tags == {}

    def test_to_audit_tags_with_feature_id(self):
        """to_audit_tags should include feature_id when set."""
        ctx = LLMRequestContext(feature_id="smart-routing-v3")

        tags = ctx.to_audit_tags()

        assert tags == {"feature_id": "smart-routing-v3"}

    def test_to_audit_tags_with_experiment_id(self):
        """to_audit_tags should include experiment_id when set."""
        ctx = LLMRequestContext(experiment_id="exp-2024-q1")

        tags = ctx.to_audit_tags()

        assert tags == {"experiment_id": "exp-2024-q1"}

    def test_to_audit_tags_with_budget_namespace(self):
        """to_audit_tags should include budget_namespace when set."""
        ctx = LLMRequestContext(budget_namespace="support-team")

        tags = ctx.to_audit_tags()

        assert tags == {"budget_namespace": "support-team"}

    def test_to_audit_tags_with_multiple_fields(self):
        """to_audit_tags should include all audit-relevant fields."""
        ctx = LLMRequestContext(
            tenant_id="acme-corp",  # not in audit tags
            feature_id="smart-routing-v3",
            experiment_id="exp-2024-q1",
            budget_namespace="support-team",
        )

        tags = ctx.to_audit_tags()

        assert tags == {
            "feature_id": "smart-routing-v3",
            "experiment_id": "exp-2024-q1",
            "budget_namespace": "support-team",
        }
        # tenant_id is not in audit tags (it's a first-class envelope field)
        assert "tenant_id" not in tags

    def test_to_audit_tags_omits_none_values(self):
        """to_audit_tags should only include non-None fields."""
        ctx = LLMRequestContext(
            feature_id="smart-routing-v3",
            experiment_id=None,
            budget_namespace="support-team",
        )

        tags = ctx.to_audit_tags()

        assert tags == {
            "feature_id": "smart-routing-v3",
            "budget_namespace": "support-team",
        }
        assert "experiment_id" not in tags
