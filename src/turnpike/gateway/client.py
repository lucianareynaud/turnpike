"""Provider-agnostic gateway client for LLM calls.

ARCHITECTURAL ROLE
──────────────────
This module is the single choke point for all LLM provider interactions.
Every call to an LLM in this application goes through ``call_llm()`` or
``call_llm_stream()``. No route or service ever imports a provider SDK
directly.

PROVIDER ABSTRACTION
─────────────────────
All provider-specific logic is encapsulated behind ``ProviderBase`` in
``turnpike.gateway.provider``. This module consumes the base class — it
does not know whether the provider is OpenAI, Anthropic, Google, or a
custom implementation.

OTEL TRACING DESIGN
────────────────────
``call_llm()`` creates one OTel span that wraps the entire LLM operation,
including retries. This span is a child of the HTTP request span created
by FastAPIInstrumentor:

  HTTP POST /summarize       (your framework, kind=SERVER)
    └── chat gpt-4o-mini      (this module, kind=CLIENT)

SPAN LIFECYCLE
──────────────
  START  →  set request attributes (route, tier, model, max_output_tokens)
  SUCCESS →  set usage attributes (tokens_in, tokens_out, cost, cache_hit)
             leave span status UNSET (OTel convention: no error)
  ERROR  →  record_exception() + StatusCode.ERROR + error.type attribute
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

__all__ = ["GatewayResult", "StreamChunk", "GatewayStream", "call_llm", "call_llm_stream"]

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from turnpike.context import LLMRequestContext
from turnpike.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    CostSource,
    EnvelopeStatus,
    LLMRequestEnvelope,
)
from turnpike.gateway.cost_model import estimate_cost
from turnpike.gateway.policies import (
    get_model_for_tier,
    get_route_policy,
)
from turnpike.gateway.provider import ProviderBase, get_provider
from turnpike.gateway.semconv import (
    ATTR_GEN_AI_OPERATION_NAME,
    ATTR_GEN_AI_REQUEST_MAX_TOKENS,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_SYSTEM,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    VAL_GEN_AI_OPERATION_CHAT,
    resolve_attrs,
)
from turnpike.gateway.telemetry import emit
from turnpike.semconv import (
    ATTR_TURNPIKE_CACHE_ENABLED,
    ATTR_TURNPIKE_CACHE_HIT,
    ATTR_TURNPIKE_CALLER_ID,
    ATTR_TURNPIKE_ERROR_CATEGORY,
    ATTR_TURNPIKE_ESTIMATED_COST_USD,
    ATTR_TURNPIKE_MODEL_TIER,
    ATTR_TURNPIKE_PROVIDER,
    ATTR_TURNPIKE_REQUEST_ID,
    ATTR_TURNPIKE_RETRY_ATTEMPTS_ALLOWED,
    ATTR_TURNPIKE_ROUTE,
    ATTR_TURNPIKE_SESSION_ID,
    ATTR_TURNPIKE_TASK_ID,
    ATTR_TURNPIKE_TENANT_ID,
    ATTR_TURNPIKE_USE_CASE,
)

ModelTier = str
GatewayRouteName = str
Message = dict[str, Any]

_tracer = trace.get_tracer(__name__, tracer_provider=None)


@dataclass(frozen=True)
class GatewayResult:
    """Structured result returned by the gateway for each LLM call.

    Frozen dataclass — immutable by design.
    """

    text: str
    selected_model: str
    request_id: str
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    cache_hit: bool
    finish_reason: str | None = None
    envelope: LLMRequestEnvelope | None = None


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    delta: str
    finish_reason: str | None = None


class GatewayStream:
    """Async iterator for streaming LLM responses.

    Yields ``StreamChunk`` objects as tokens arrive.  After iteration
    completes, the ``result`` property exposes the full ``GatewayResult``
    with envelope, cost, and token counts.

    Usage::

        stream = call_llm_stream(prompt="...", model_tier="cheap", route_name="/route")
        async for chunk in stream:
            print(chunk.delta, end="", flush=True)
        print(stream.result.envelope.to_dict())
    """

    def __init__(self, aiter: AsyncIterator[StreamChunk]) -> None:
        self._aiter = aiter
        self._result: GatewayResult | None = None

    def _set_result(self, result: GatewayResult) -> None:
        self._result = result

    @property
    def result(self) -> GatewayResult:
        """Full result with envelope — available after iteration completes."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self._aiter


def _resolve_messages(
    prompt: str | None,
    messages: list[Message] | None,
) -> list[Message]:
    """Convert prompt/messages args to a canonical message list."""
    if messages is not None:
        return messages
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    raise ValueError("Either 'prompt' or 'messages' must be provided.")


def _resolve_context(
    context: LLMRequestContext | None,
    metadata: dict[str, Any] | None,
) -> tuple[LLMRequestContext, str, str | None, str | None, str | None, str | None, dict[str, str]]:
    """Extract attribution fields from context or metadata."""
    effective = context if context is not None else LLMRequestContext.from_metadata(metadata)
    tenant_id = effective.tenant_id or "default"
    return (
        effective,
        tenant_id,
        effective.caller_id,
        effective.use_case,
        effective.session_id,
        effective.task_id,
        effective.to_audit_tags(),
    )


def _set_span_request_attrs(
    span: trace.Span,
    *,
    provider: ProviderBase,
    selected_model: str,
    max_output_tokens: int,
    route_name: str,
    model_tier: str,
    request_id: str,
    tenant_id: str,
    caller_id: str | None,
    use_case: str | None,
    session_id: str | None,
    task_id: str | None,
    audit_tags: dict[str, str],
    policy_retry_attempts: int,
    policy_cache_enabled: bool,
) -> None:
    """Set all request-phase OTel span attributes."""
    request_attrs = resolve_attrs(
        {
            ATTR_GEN_AI_SYSTEM: provider.provider_name,
            ATTR_GEN_AI_OPERATION_NAME: VAL_GEN_AI_OPERATION_CHAT,
            ATTR_GEN_AI_REQUEST_MODEL: selected_model,
            ATTR_GEN_AI_REQUEST_MAX_TOKENS: max_output_tokens,
        }
    )
    for key, value in request_attrs.items():
        span.set_attribute(key, value)

    span.set_attribute(ATTR_TURNPIKE_ROUTE, route_name)
    span.set_attribute(ATTR_TURNPIKE_MODEL_TIER, model_tier)
    span.set_attribute(ATTR_TURNPIKE_REQUEST_ID, request_id)
    span.set_attribute(ATTR_TURNPIKE_TENANT_ID, tenant_id)
    if caller_id is not None:
        span.set_attribute(ATTR_TURNPIKE_CALLER_ID, caller_id)
    if use_case is not None:
        span.set_attribute(ATTR_TURNPIKE_USE_CASE, use_case)
    if session_id is not None:
        span.set_attribute(ATTR_TURNPIKE_SESSION_ID, session_id)
    if task_id is not None:
        span.set_attribute(ATTR_TURNPIKE_TASK_ID, task_id)
    for tag_key, tag_value in audit_tags.items():
        span.set_attribute(f"turnpike.audit.{tag_key}", tag_value)
    span.set_attribute(ATTR_TURNPIKE_RETRY_ATTEMPTS_ALLOWED, policy_retry_attempts)
    span.set_attribute(ATTR_TURNPIKE_CACHE_ENABLED, policy_cache_enabled)
    span.set_attribute(ATTR_TURNPIKE_PROVIDER, provider.provider_name)


async def call_llm(
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    model_tier: ModelTier,
    route_name: GatewayRouteName,
    metadata: dict[str, Any] | None = None,
    context: LLMRequestContext | None = None,
) -> GatewayResult:
    """Execute one LLM call through the gateway.

    This is the primary entry point for non-streaming LLM calls.
    All parameters are keyword-only.

    Args:
        prompt:      Convenience shorthand — converted to
                     ``[{"role": "user", "content": prompt}]``.
        messages:    Full message list (e.g. multi-turn conversation).
                     Takes precedence over ``prompt`` when both are given.
        model_tier:  Logical tier (e.g. ``"cheap"``, ``"expensive"``).
        route_name:  Gateway route identifier.
        metadata:    Optional route-specific key-values for telemetry (legacy).
        context:     Optional structured request context for attribution (preferred).

    Returns:
        GatewayResult with response text, model, token counts, cost.

    Raises:
        ValueError: If neither prompt nor messages is provided, or provider
                    SDK is not installed.
        Exception: Provider-specific exceptions after all retries.
    """
    effective_messages = _resolve_messages(prompt, messages)

    request_id = str(uuid.uuid4())
    policy = get_route_policy(route_name)
    selected_model = get_model_for_tier(route_name, model_tier)
    provider = get_provider(policy.provider_name)

    (_, tenant_id, caller_id, use_case, session_id, task_id, audit_tags) = _resolve_context(
        context, metadata
    )

    telemetry_metadata = dict(metadata or {})
    telemetry_metadata["selected_model"] = selected_model

    span_name = f"{VAL_GEN_AI_OPERATION_CHAT} {selected_model}"

    with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
        _set_span_request_attrs(
            span,
            provider=provider,
            selected_model=selected_model,
            max_output_tokens=policy.max_output_tokens,
            route_name=route_name,
            model_tier=model_tier,
            request_id=request_id,
            tenant_id=tenant_id,
            caller_id=caller_id,
            use_case=use_case,
            session_id=session_id,
            task_id=task_id,
            audit_tags=audit_tags,
            policy_retry_attempts=policy.retry_attempts,
            policy_cache_enabled=policy.cache_enabled,
        )

        start_time = time.perf_counter()

        try:
            text, tokens_in, tokens_out, retries_used, finish_reason = await _call_provider(
                provider=provider,
                messages=effective_messages,
                model=selected_model,
                max_output_tokens=policy.max_output_tokens,
                retry_attempts=policy.retry_attempts,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            cost_usd = estimate_cost(selected_model, tokens_in, tokens_out)

            usage_attrs = resolve_attrs(
                {
                    ATTR_GEN_AI_USAGE_INPUT_TOKENS: tokens_in,
                    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: tokens_out,
                }
            )
            for key, value in usage_attrs.items():
                span.set_attribute(key, value)

            span.set_attribute(ATTR_TURNPIKE_ESTIMATED_COST_USD, cost_usd)
            span.set_attribute(ATTR_TURNPIKE_CACHE_HIT, False)

            envelope = LLMRequestEnvelope(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                request_id=request_id,
                tenant_id=tenant_id,
                caller_id=caller_id,
                use_case=use_case,
                session_id=session_id,
                task_id=task_id,
                route=route_name,
                provider_selected=provider.provider_name,
                model_selected=selected_model,
                model_tier=model_tier,
                routing_decision=telemetry_metadata.get("routing_decision"),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_total=tokens_in + tokens_out,
                estimated_cost_usd=cost_usd,
                cost_source=CostSource.ESTIMATED_LOCAL_SNAPSHOT,
                latency_ms=latency_ms,
                status=EnvelopeStatus.OK,
                retry_count=retries_used,
                streaming=False,
                finish_reason=finish_reason,
                cache_hit=False,
                audit_tags=audit_tags,
            )

            emit(
                request_id=request_id,
                route=route_name,
                provider=provider.provider_name,
                model=selected_model,
                latency_ms=latency_ms,
                status="success",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                estimated_cost_usd=cost_usd,
                cache_hit=False,
                schema_valid=True,
                error_type=None,
                metadata=telemetry_metadata,
                envelope=envelope,
            )

            return GatewayResult(
                text=text,
                selected_model=selected_model,
                request_id=request_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                estimated_cost_usd=cost_usd,
                cache_hit=False,
                finish_reason=finish_reason,
                envelope=envelope,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            error_type = provider.categorize_error(exc)

            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute(ATTR_TURNPIKE_ERROR_CATEGORY, error_type)

            envelope = LLMRequestEnvelope(
                schema_version=ENVELOPE_SCHEMA_VERSION,
                request_id=request_id,
                tenant_id=tenant_id,
                caller_id=caller_id,
                use_case=use_case,
                session_id=session_id,
                task_id=task_id,
                route=route_name,
                provider_selected=provider.provider_name,
                model_selected=selected_model,
                model_tier=model_tier,
                routing_decision=telemetry_metadata.get("routing_decision"),
                tokens_in=0,
                tokens_out=0,
                tokens_total=0,
                estimated_cost_usd=0.0,
                cost_source=CostSource.DEGRADED_UNKNOWN,
                latency_ms=latency_ms,
                status=EnvelopeStatus.ERROR,
                error_type=error_type,
                streaming=False,
                cache_hit=False,
                audit_tags=audit_tags,
            )

            emit(
                request_id=request_id,
                route=route_name,
                provider=provider.provider_name,
                model=selected_model,
                latency_ms=latency_ms,
                status="error",
                tokens_in=0,
                tokens_out=0,
                estimated_cost_usd=0.0,
                cache_hit=False,
                schema_valid=True,
                error_type=error_type,
                metadata=telemetry_metadata,
                envelope=envelope,
            )

            raise


def call_llm_stream(
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    model_tier: ModelTier,
    route_name: GatewayRouteName,
    metadata: dict[str, Any] | None = None,
    context: LLMRequestContext | None = None,
) -> GatewayStream:
    """Stream an LLM call through the gateway.

    Returns a ``GatewayStream`` — an async iterator that yields
    ``StreamChunk`` objects. After full consumption, ``stream.result``
    provides the complete ``GatewayResult`` with envelope and cost.
    All parameters are keyword-only.

    Args:
        prompt:      Convenience shorthand for a single user message.
        messages:    Full message list. Takes precedence over ``prompt``.
        model_tier:  Logical tier (e.g. ``"cheap"``, ``"expensive"``).
        route_name:  Gateway route identifier.
        metadata:    Optional route-specific key-values for telemetry.
        context:     Optional structured request context for attribution.

    Returns:
        GatewayStream that yields StreamChunk objects.
    """
    effective_messages = _resolve_messages(prompt, messages)

    request_id = str(uuid.uuid4())
    policy = get_route_policy(route_name)
    selected_model = get_model_for_tier(route_name, model_tier)
    provider = get_provider(policy.provider_name)

    (_, tenant_id, caller_id, use_case, session_id, task_id, audit_tags) = _resolve_context(
        context, metadata
    )

    telemetry_metadata = dict(metadata or {})
    telemetry_metadata["selected_model"] = selected_model

    async def _generate() -> AsyncIterator[StreamChunk]:
        span_name = f"{VAL_GEN_AI_OPERATION_CHAT} {selected_model}"

        with _tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            _set_span_request_attrs(
                span,
                provider=provider,
                selected_model=selected_model,
                max_output_tokens=policy.max_output_tokens,
                route_name=route_name,
                model_tier=model_tier,
                request_id=request_id,
                tenant_id=tenant_id,
                caller_id=caller_id,
                use_case=use_case,
                session_id=session_id,
                task_id=task_id,
                audit_tags=audit_tags,
                policy_retry_attempts=policy.retry_attempts,
                policy_cache_enabled=policy.cache_enabled,
            )

            start_time = time.perf_counter()
            ttft: float | None = None
            tokens_in = 0
            tokens_out = 0
            finish_reason: str | None = None
            accumulated_text = ""

            try:
                async for event in provider.stream(
                    effective_messages, selected_model, policy.max_output_tokens
                ):
                    if event.delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - start_time) * 1000.0
                        accumulated_text += event.delta
                        yield StreamChunk(delta=event.delta)

                    if event.tokens_in is not None:
                        tokens_in = event.tokens_in
                    if event.tokens_out is not None:
                        tokens_out = event.tokens_out
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason
                        yield StreamChunk(delta="", finish_reason=finish_reason)

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                cost_usd = estimate_cost(selected_model, tokens_in, tokens_out)

                usage_attrs = resolve_attrs(
                    {
                        ATTR_GEN_AI_USAGE_INPUT_TOKENS: tokens_in,
                        ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: tokens_out,
                    }
                )
                for key, value in usage_attrs.items():
                    span.set_attribute(key, value)
                span.set_attribute(ATTR_TURNPIKE_ESTIMATED_COST_USD, cost_usd)
                span.set_attribute(ATTR_TURNPIKE_CACHE_HIT, False)

                envelope = LLMRequestEnvelope(
                    schema_version=ENVELOPE_SCHEMA_VERSION,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    caller_id=caller_id,
                    use_case=use_case,
                    session_id=session_id,
                    task_id=task_id,
                    route=route_name,
                    provider_selected=provider.provider_name,
                    model_selected=selected_model,
                    model_tier=model_tier,
                    routing_decision=telemetry_metadata.get("routing_decision"),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    tokens_total=tokens_in + tokens_out,
                    estimated_cost_usd=cost_usd,
                    cost_source=CostSource.ESTIMATED_LOCAL_SNAPSHOT,
                    latency_ms=latency_ms,
                    time_to_first_token_ms=ttft,
                    status=EnvelopeStatus.OK,
                    retry_count=0,
                    streaming=True,
                    finish_reason=finish_reason,
                    cache_hit=False,
                    audit_tags=audit_tags,
                )

                emit(
                    request_id=request_id,
                    route=route_name,
                    provider=provider.provider_name,
                    model=selected_model,
                    latency_ms=latency_ms,
                    status="success",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    estimated_cost_usd=cost_usd,
                    cache_hit=False,
                    schema_valid=True,
                    error_type=None,
                    metadata=telemetry_metadata,
                    envelope=envelope,
                )

                gateway_stream._set_result(
                    GatewayResult(
                        text=accumulated_text,
                        selected_model=selected_model,
                        request_id=request_id,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        estimated_cost_usd=cost_usd,
                        cache_hit=False,
                        finish_reason=finish_reason,
                        envelope=envelope,
                    )
                )

            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                error_type = provider.categorize_error(exc)

                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute(ATTR_TURNPIKE_ERROR_CATEGORY, error_type)

                emit(
                    request_id=request_id,
                    route=route_name,
                    provider=provider.provider_name,
                    model=selected_model,
                    latency_ms=latency_ms,
                    status="error",
                    tokens_in=0,
                    tokens_out=0,
                    estimated_cost_usd=0.0,
                    cache_hit=False,
                    schema_valid=True,
                    error_type=error_type,
                    metadata=telemetry_metadata,
                )

                raise

    gateway_stream = GatewayStream(_generate())
    return gateway_stream


async def _call_provider(
    provider: ProviderBase,
    messages: list[Message],
    model: str,
    max_output_tokens: int,
    retry_attempts: int,
) -> tuple[str, int, int, int, str | None]:
    """Call the provider with bounded exponential backoff retry.

    Retry decision delegated to ``provider.is_retryable()``.

    Returns:
        Tuple of (response_text, tokens_in, tokens_out, retries_used, finish_reason).
        retries_used is 0 on first-attempt success, 1 after one retry, etc.
    """
    last_exception: Exception | None = None

    for attempt in range(retry_attempts + 1):
        try:
            response = await provider.complete(messages, model, max_output_tokens)
            return (
                response.text,
                response.tokens_in,
                response.tokens_out,
                attempt,
                response.finish_reason,
            )

        except Exception as exc:
            last_exception = exc

            if provider.is_retryable(exc) and attempt < retry_attempts:
                await asyncio.sleep(2**attempt)
                continue

            break

    raise last_exception or RuntimeError("Provider call failed with no exception captured")
