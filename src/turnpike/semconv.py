"""
Turnpike semantic conventions — project-specific attribute names.

This module defines the `turnpike.*` namespace constants for attributes
that are not covered by official OpenTelemetry semantic conventions.

Use official `gen_ai.*` semconv from gateway/semconv.py for standard fields.
"""

# Schema versioning
ATTR_TURNPIKE_SCHEMA_VERSION = "turnpike.schema_version"

# Economics
ATTR_TURNPIKE_COST_SOURCE = "turnpike.cost_source"
ATTR_TURNPIKE_ESTIMATED_COST_USD = "turnpike.estimated_cost_usd"
ATTR_TURNPIKE_TOKENS_TOTAL = "turnpike.tokens_total"

# Identity and context
ATTR_TURNPIKE_REQUEST_ID = "turnpike.request_id"
ATTR_TURNPIKE_TENANT_ID = "turnpike.tenant_id"
ATTR_TURNPIKE_CALLER_ID = "turnpike.caller_id"
ATTR_TURNPIKE_USE_CASE = "turnpike.use_case"
ATTR_TURNPIKE_SESSION_ID = "turnpike.session_id"
ATTR_TURNPIKE_TASK_ID = "turnpike.task_id"
ATTR_TURNPIKE_EVENT_TYPE = "turnpike.event_type"
ATTR_TURNPIKE_ROUTE = "turnpike.route"
ATTR_TURNPIKE_RUNTIME_MODE = "turnpike.runtime_mode"

# Model selection and routing
ATTR_TURNPIKE_MODEL_TIER = "turnpike.model_tier"
ATTR_TURNPIKE_ROUTING_DECISION = "turnpike.routing_decision"
ATTR_TURNPIKE_ROUTING_REASON = "turnpike.routing_reason"

# Reliability
ATTR_TURNPIKE_STATUS = "turnpike.status"
ATTR_TURNPIKE_ERROR_TYPE = "turnpike.error_type"
ATTR_TURNPIKE_LATENCY_MS = "turnpike.latency_ms"
ATTR_TURNPIKE_RETRY_COUNT = "turnpike.retry_count"
ATTR_TURNPIKE_FALLBACK_TRIGGERED = "turnpike.fallback_triggered"
ATTR_TURNPIKE_FALLBACK_REASON = "turnpike.fallback_reason"
ATTR_TURNPIKE_CIRCUIT_STATE = "turnpike.circuit_state"

# Governance
ATTR_TURNPIKE_POLICY_INPUT_CLASS = "turnpike.policy_input_class"
ATTR_TURNPIKE_POLICY_DECISION = "turnpike.policy_decision"
ATTR_TURNPIKE_POLICY_MODE = "turnpike.policy_mode"
ATTR_TURNPIKE_REDACTION_APPLIED = "turnpike.redaction_applied"
ATTR_TURNPIKE_PII_DETECTED = "turnpike.pii_detected"

# Cache and evaluation
ATTR_TURNPIKE_CACHE_ELIGIBLE = "turnpike.cache_eligible"
ATTR_TURNPIKE_CACHE_STRATEGY = "turnpike.cache_strategy"
ATTR_TURNPIKE_CACHE_HIT = "turnpike.cache_hit"
ATTR_TURNPIKE_CACHE_KEY_FINGERPRINT = "turnpike.cache_key_fingerprint"
ATTR_TURNPIKE_CACHE_KEY_ALGORITHM = "turnpike.cache_key_algorithm"
ATTR_TURNPIKE_CACHE_LOOKUP_CONFIDENCE = "turnpike.cache_lookup_confidence"
ATTR_TURNPIKE_EVAL_HOOKS = "turnpike.eval_hooks"

# Gateway-specific (span attributes used in client.py)
ATTR_TURNPIKE_RETRY_ATTEMPTS_ALLOWED = "turnpike.retry_attempts_allowed"
ATTR_TURNPIKE_CACHE_ENABLED = "turnpike.cache_enabled"
ATTR_TURNPIKE_PROVIDER = "turnpike.provider"
ATTR_TURNPIKE_ERROR_CATEGORY = "turnpike.error_category"

# Metric instrument names
METRIC_TURNPIKE_COST_USD = "turnpike.estimated_cost_usd"
METRIC_TURNPIKE_REQUESTS = "turnpike.requests"
