"""Structured request context for economic attribution and audit trails.

This module provides LLMRequestContext, a small immutable type that carries
attribution metadata through the gateway runtime without coupling to framework
types or policy engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["LLMRequestContext"]


@dataclass(frozen=True)
class LLMRequestContext:
    """Structured context for LLM request attribution and audit.

    All fields are optional. When absent, the runtime uses safe defaults.
    This type is immutable to prevent accidental mutation during request flow.

    Fields:
        tenant_id: Logical tenant identifier for multi-tenant cost attribution
        caller_id: Service or user identifier for caller attribution
        use_case: Business use case label (e.g. "customer-support", "code-review")
        session_id: Logical session identifier for multi-turn correlation
        task_id: Task or sub-task identifier within a session
        feature_id: Feature flag or experiment variant identifier
        experiment_id: A/B test or experiment identifier
        budget_namespace: Cost budget namespace for spend tracking
    """

    tenant_id: str | None = None
    caller_id: str | None = None
    use_case: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    feature_id: str | None = None
    experiment_id: str | None = None
    budget_namespace: str | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> LLMRequestContext:
        """Create context from legacy metadata dict for backward compatibility.

        Extracts known attribution fields from metadata. Unknown fields are ignored.
        This bridge allows existing callers using metadata to benefit from structured
        context without code changes.

        Args:
            metadata: Legacy metadata dict, may be None

        Returns:
            LLMRequestContext with fields extracted from metadata
        """
        if metadata is None:
            return cls()

        return cls(
            tenant_id=metadata.get("tenant_id"),
            caller_id=metadata.get("caller_id"),
            use_case=metadata.get("use_case"),
            session_id=metadata.get("session_id"),
            task_id=metadata.get("task_id"),
            feature_id=metadata.get("feature_id"),
            experiment_id=metadata.get("experiment_id"),
            budget_namespace=metadata.get("budget_namespace"),
        )

    def to_audit_tags(self) -> dict[str, str]:
        """Project context fields into audit tags for envelope extension.

        Returns a dict of non-None context fields formatted as audit tags.
        These tags are deterministic and derived only from structured context,
        never from arbitrary caller input.

        Returns:
            Dict of audit tag key-value pairs (empty if no fields set)
        """
        tags: dict[str, str] = {}

        if self.session_id is not None:
            tags["session_id"] = self.session_id
        if self.task_id is not None:
            tags["task_id"] = self.task_id
        if self.feature_id is not None:
            tags["feature_id"] = self.feature_id
        if self.experiment_id is not None:
            tags["experiment_id"] = self.experiment_id
        if self.budget_namespace is not None:
            tags["budget_namespace"] = self.budget_namespace

        return tags
