[![PyPI](https://img.shields.io/pypi/v/turnpike)](https://pypi.org/project/turnpike/)
[![CI](https://github.com/lucianareynaud/turnpike/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianareynaud/turnpike/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

# Turnpike

A small, typed Python package for LLM cost attribution. Every LLM call
produces one `LLMRequestEnvelope` — a frozen dataclass that records who
called what model, how many tokens it consumed, what it cost, whether it
retried, and under which governance policy it ran.

```json
{
  "schema_version": "0.1.0",
  "request_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "tenant_id": "acme",
  "caller_id": "billing-svc",
  "session_id": "sess-0042",
  "route": "/summarize",
  "provider_selected": "openai",
  "model_selected": "gpt-4o-mini",
  "model_tier": "cheap",
  "tokens_in": 150,
  "tokens_out": 80,
  "tokens_total": 230,
  "estimated_cost_usd": 0.00007,
  "cost_source": "estimated_local_snapshot",
  "latency_ms": 342.5,
  "time_to_first_token_ms": 187.2,
  "status": "ok",
  "retry_count": 0,
  "streaming": true,
  "finish_reason": "stop",
  "cache_hit": false
}
```

The envelope flows to OpenTelemetry metrics, JSONL files, or both. The
gateway, providers, and OTel instrumentation exist to populate it.

## Why Turnpike

Distributed tracing was built for request-reply microservices. LLM systems
break that model: calls fan out non-deterministically, costs compound
across retries and fallbacks, and a single "request" may span multiple
providers.

Existing observability tools solve *visibility*. Turnpike solves
*accountability* — who spent what, on which model, under which policy.

| Concern | Tracing alone | Turnpike |
|---|---|---|
| Token counts | Manual span attributes | `tokens_in` / `tokens_out` / `tokens_total` |
| Cost attribution | Not modeled | `estimated_cost_usd` + `cost_source` provenance |
| Multi-tenant billing | Custom baggage | `tenant_id` + `caller_id` + `budget_namespace` |
| Policy enforcement | Out of scope | `policy_decision` + `redaction_applied` |
| Retry / fallback | Span count heuristics | `retry_count` + `fallback_triggered` + `circuit_state` |
| Streaming latency | No convention | `time_to_first_token_ms` + `finish_reason` |

**What Turnpike is not.** It is not a proxy (like LiteLLM), not a
dashboard (like LangSmith), and not a prompt management tool. It is a
typed envelope and telemetry primitives layer that sits between your LLM
calls and your observability backend.

Read more: [Why Distributed Tracing Doesn't Model LLM Agent Observability](https://lucianareynaud.medium.com/why-distributed-tracing-doesnt-model-llm-agent-observability-c376bd3ac2fc)

## Installation

```bash
pip install turnpike              # core envelope + OTel instrumentation
pip install turnpike[openai]      # + OpenAI provider
pip install turnpike[anthropic]   # + Anthropic provider
pip install turnpike[all]         # all providers
```

## Quick Start

### Standalone envelope (no gateway, no provider SDK)

The envelope is a frozen dataclass. If you already have your own LLM
integration, construct envelopes directly and feed them to your
observability pipeline:

```python
from turnpike import LLMRequestEnvelope, EnvelopeStatus, CostSource

envelope = LLMRequestEnvelope(
    schema_version="0.1.0",
    request_id="req-abc",
    tenant_id="acme",
    route="/summarize",
    provider_selected="anthropic",
    model_selected="claude-sonnet-4-20250514",
    tokens_in=1200,
    tokens_out=350,
    tokens_total=1550,
    estimated_cost_usd=0.00885,
    cost_source=CostSource.ESTIMATED_LOCAL_SNAPSHOT,
    status=EnvelopeStatus.OK,
    streaming=False,
    finish_reason="end_turn",
    session_id="sess-0042",
    task_id="planning-step",
)

event = envelope.to_dict()  # JSON-safe dict, enums serialized as strings
```

Deserialize from JSONL or wire format:

```python
from turnpike import LLMRequestEnvelope

reconstructed = LLMRequestEnvelope.from_dict(event)
```

### Gateway call (prompt shorthand)

Register a route policy, then call any LLM through the gateway. Every
call returns a `GatewayResult` carrying token counts, cost, and the full
envelope:

```python
import asyncio
from turnpike import (
    call_llm, LLMRequestContext,
    register_route_policy, RoutePolicy,
)

register_route_policy("/summarize", RoutePolicy(
    max_output_tokens=500,
    retry_attempts=2,
    cache_enabled=False,
    model_for_tier={"cheap": "gpt-4o-mini", "expensive": "gpt-4o"},
    provider_name="openai",
))

ctx = LLMRequestContext(
    tenant_id="acme",
    caller_id="billing-svc",
    session_id="sess-0042",
    task_id="invoice-summarization",
    budget_namespace="finance-team",
)

async def main():
    result = await call_llm(
        prompt="Summarise this invoice.",
        model_tier="cheap",
        route_name="/summarize",
        context=ctx,
    )
    print(result.estimated_cost_usd)          # 0.00007
    print(result.finish_reason)               # "stop"
    print(result.envelope.tenant_id)          # "acme"
    print(result.envelope.retry_count)        # 0
    print(result.envelope.cost_source)        # CostSource.ESTIMATED_LOCAL_SNAPSHOT

asyncio.run(main())
```

### Gateway call (multi-turn messages)

Pass a full conversation history using `messages` instead of `prompt`:

```python
import asyncio
from turnpike import call_llm, LLMRequestContext

ctx = LLMRequestContext(tenant_id="acme", caller_id="billing-svc")

async def main():
    result = await call_llm(
        messages=[
            {"role": "system", "content": "You summarise invoices."},
            {"role": "user", "content": "Here is invoice #1234..."},
            {"role": "assistant", "content": "Invoice #1234 totals $5,200."},
            {"role": "user", "content": "Break down the line items."},
        ],
        model_tier="expensive",
        route_name="/summarize",
        context=ctx,
    )
    print(result.content)

asyncio.run(main())
```

### Streaming

Stream responses token-by-token. The `GatewayStream` yields `StreamChunk`
objects; after iteration, `.result` holds the full `GatewayResult` with
envelope, cost, and TTFT:

```python
import asyncio
from turnpike import call_llm_stream, LLMRequestContext

ctx = LLMRequestContext(tenant_id="acme", caller_id="billing-svc")

async def main():
    stream = call_llm_stream(
        prompt="Summarise this invoice.",
        model_tier="cheap",
        route_name="/summarize",
        context=ctx,
    )
    async for chunk in stream:
        print(chunk.delta, end="", flush=True)

    print(stream.result.envelope.time_to_first_token_ms)  # 187.2
    print(stream.result.envelope.streaming)                # True
    print(stream.result.envelope.finish_reason)            # "stop"
    print(stream.result.estimated_cost_usd)                # 0.00007

asyncio.run(main())
```

## What Turnpike Produces

### OTel metrics (4 instruments)

Every `call_llm()` records metrics into whatever OTLP-compatible backend
you configure (Grafana, Datadog, Prometheus, New Relic):

| Instrument | Type | Unit | Purpose |
|---|---|---|---|
| `gen_ai.client.token.usage` | Histogram | `{token}` | Input and output token counts per call |
| `gen_ai.client.operation.duration` | Histogram | `s` | Wall-clock latency including retries |
| `turnpike.estimated_cost_usd` | Counter | `USD` | Cumulative cost by route, model, and tier |
| `turnpike.requests` | Counter | `{request}` | Request volume by route, tier, and status |

Metric attributes follow the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) with `OTEL_SEMCONV_STABILITY_OPT_IN` dual-emission support.

### OTel spans

Each `call_llm()` creates a `CLIENT` span as a child of the caller's
current span context:

```
HTTP POST /summarize                          (your framework, kind=SERVER)
  └── chat gpt-4o-mini                        (turnpike, kind=CLIENT)
        attributes:
          gen_ai.system = "openai"
          gen_ai.request.model = "gpt-4o-mini"
          gen_ai.usage.input_tokens = 150
          gen_ai.usage.output_tokens = 80
          turnpike.route = "/summarize"
          turnpike.tenant_id = "acme"
          turnpike.session_id = "sess-0042"
          turnpike.estimated_cost_usd = 0.00007
```

### JSONL telemetry file

Every call also appends one JSON line to `~/.turnpike/telemetry.jsonl`
(configurable via `TURNPIKE_TELEMETRY_PATH`). This file is consumable by
`jq`, DuckDB, pandas, or any JSONL-aware tool.

> **Note on status values:** The JSONL telemetry layer uses `"success"` /
> `"error"` as outcome values. The `LLMRequestEnvelope.status` field uses
> its own `EnvelopeStatus` enum (`ok`, `cached`, `error`, `degraded`,
> `denied`). When an envelope is serialized to JSONL, `"ok"` is mapped to
> `"success"` for compatibility with reporting tools. These are distinct
> semantic layers — do not assume they are interchangeable.

## Envelope Field Reference

`LLMRequestEnvelope` is a frozen dataclass with fields across six
semantic blocks. Four fields are required; the rest default to `None` or
safe enum values.

### Identity and context

| Field | Type | Required | Purpose |
|---|---|---|---|
| `schema_version` | `str` | yes | Envelope schema version |
| `request_id` | `str` | yes | Unique identifier for this LLM call |
| `tenant_id` | `str` | yes | Tenant for multi-tenant cost attribution |
| `route` | `str` | yes | Logical route or operation name |
| `trace_id` | `str \| None` | | OTel trace ID for cross-service correlation |
| `span_id` | `str \| None` | | OTel span ID |
| `caller_id` | `str \| None` | | Service or user that initiated the call |
| `use_case` | `str \| None` | | Business use case label |
| `session_id` | `str \| None` | | Session identifier for multi-turn correlation |
| `task_id` | `str \| None` | | Task or sub-task identifier within a session |

### Model selection

| Field | Type | Default | Purpose |
|---|---|---|---|
| `provider_requested` | `str \| None` | | Provider the caller asked for |
| `model_requested` | `str \| None` | | Model the caller asked for |
| `provider_selected` | `str \| None` | | Provider actually used |
| `model_selected` | `str \| None` | | Model actually used |
| `model_tier` | `str \| None` | | Logical tier (e.g. `"cheap"`, `"expensive"`) |
| `routing_decision` | `str \| None` | | How the model was selected |
| `routing_reason` | `str \| None` | | Human-readable routing rationale |

### Economics

| Field | Type | Default | Purpose |
|---|---|---|---|
| `tokens_in` | `int \| None` | | Input token count |
| `tokens_out` | `int \| None` | | Output token count |
| `tokens_total` | `int \| None` | | Sum of input + output tokens |
| `estimated_cost_usd` | `float \| None` | | Estimated cost in USD |
| `cost_source` | `CostSource` | `DEGRADED_UNKNOWN` | Provenance of cost estimation |

### Reliability

| Field | Type | Default | Purpose |
|---|---|---|---|
| `latency_ms` | `float \| None` | | Wall-clock duration including retries |
| `time_to_first_token_ms` | `float \| None` | | Time to first streamed token |
| `status` | `EnvelopeStatus` | `OK` | Terminal status: `ok`, `cached`, `error`, `degraded`, `denied` |
| `error_type` | `str \| None` | | Categorised error type |
| `retry_count` | `int \| None` | | Number of retry attempts |
| `fallback_triggered` | `bool \| None` | | Whether a fallback model was used |
| `fallback_reason` | `str \| None` | | Why fallback was triggered |
| `circuit_state` | `CircuitState \| None` | | Circuit breaker state |
| `streaming` | `bool \| None` | | Whether the call used streaming |
| `finish_reason` | `str \| None` | | Provider finish reason (e.g. `stop`, `length`) |

### Governance

| Field | Type | Default | Purpose |
|---|---|---|---|
| `policy_input_class` | `str \| None` | | Classification of the input content |
| `policy_decision` | `str \| None` | | Policy engine decision |
| `policy_mode` | `str \| None` | | Policy enforcement mode |
| `redaction_applied` | `bool \| None` | | Whether PII redaction was applied |
| `pii_detected` | `bool \| None` | | Whether PII was detected |

### Cache and evaluation

| Field | Type | Default | Purpose |
|---|---|---|---|
| `cache_eligible` | `bool \| None` | | Whether this request was eligible for caching |
| `cache_strategy` | `str \| None` | | Cache strategy |
| `cache_hit` | `bool \| None` | | Whether the response came from cache |
| `cache_key_fingerprint` | `str \| None` | | Cache key hash |
| `cache_key_algorithm` | `str \| None` | | Hash algorithm used for cache key |
| `cache_lookup_confidence` | `float \| None` | | Confidence score for semantic cache matches |
| `eval_hooks` | `tuple[str, ...]` | `()` | Eval hook names that ran on this request |
| `audit_tags` | `dict[str, str]` | `{}` | Extensible key-value pairs for audit trails |

**Schema evolution:** new optional fields may appear in minor versions.
Consumers should tolerate unknown keys. Removing or renaming a field
requires a major version change.

## Top-Level API

Stable imports from `turnpike`:

```python
from turnpike import (
    # Envelope and context
    LLMRequestEnvelope, LLMRequestContext,
    EnvelopeStatus, CostSource,
    # Gateway results
    GatewayResult, GatewayStream, StreamChunk,
    # Call entrypoints
    call_llm, call_llm_stream,
    # Provider abstraction
    ProviderBase, ProviderResponse, register_provider,
    # Policy
    RoutePolicy, register_route_policy,
    # Cost
    estimate_cost, register_pricing,
    # Telemetry
    emit_event, setup_otel, shutdown_otel,
)
```

Vendor-specific providers (`OpenAIProvider`, `AnthropicProvider`) and
internal helpers (`get_provider`, `available_providers`, `get_pricing`,
`get_route_policy`, `get_model_for_tier`, `clear_route_policies`) are
importable from their submodules but are not part of the top-level public
contract.

## Architecture

```
call_llm(prompt= or messages=)  [src/turnpike/gateway/client.py]
  ├── RoutePolicy lookup (tier → model mapping)
  ├── LLMRequestContext resolution
  ├── OTel CLIENT span start
  ├── Retry loop with exponential backoff
  │     └── ProviderBase.complete(messages) → OpenAI / Anthropic / custom
  ├── estimate_cost()
  ├── LLMRequestEnvelope construction
  └── emit() → OTel metrics + JSONL artifact

call_llm_stream(prompt= or messages=)
  ├── Same setup as call_llm
  ├── ProviderBase.stream(messages) → yields ProviderStreamEvent
  ├── GatewayStream yields StreamChunk (with TTFT measurement)
  └── On completion: envelope + emit() + GatewayResult via stream.result
```

## Adding a Custom Provider

Providers are available from `turnpike.gateway.provider`. The built-in
`OpenAIProvider` and `AnthropicProvider` require their respective
optional extras (`pip install turnpike[openai]`).

```python
from turnpike import ProviderBase, ProviderResponse, register_provider, register_pricing

class MyProvider(ProviderBase):
    @property
    def provider_name(self) -> str:
        return "my-provider"

    async def complete(self, messages, model, max_output_tokens) -> ProviderResponse:
        response = await my_api_call(messages, model, max_output_tokens)
        return ProviderResponse(
            text=response.text,
            tokens_in=response.usage.input,
            tokens_out=response.usage.output,
            finish_reason=response.stop_reason,
            response_model=response.model,
        )

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, (RateLimitError, TimeoutError))

    def categorize_error(self, error: Exception) -> str:
        if isinstance(error, RateLimitError):
            return "rate_limit"
        return "unknown"

register_provider("my-provider", MyProvider())
register_pricing("my-model-v1", input_per_1m=1.00, output_per_1m=3.00)
```

The provider contract: `provider_name` and `complete()` are abstract;
`is_retryable()`, `categorize_error()`, and `stream()` have safe
defaults. New optional methods may appear in minor versions. New abstract
methods require a major version change.

## Configuration

| Environment Variable | Purpose | Default |
|---|---|---|
| `TURNPIKE_TELEMETRY_PATH` | JSONL telemetry output path | `~/.turnpike/telemetry.jsonl` |
| `OPENAI_API_KEY` | OpenAI provider auth | — |
| `ANTHROPIC_API_KEY` | Anthropic provider auth | — |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | Console fallback |
| `OTEL_SDK_DISABLED` | Disable OTel SDK (`true` for CI) | `false` |
| `OTEL_SERVICE_NAME` | Service name in OTel resource | `turnpike` |

## Versioning

This is an early release with a narrow public API. The package follows
[Semantic Versioning](https://semver.org/).

**Top-level imports** — all names in `turnpike.__all__` are covered by
SemVer. Internal module paths (`turnpike.gateway.client`, etc.) are
implementation details and may change in minor versions.

**Envelope schema** — `LLMRequestEnvelope` carries a `schema_version`
field. New optional fields may appear in minor versions. Removing or
renaming fields requires a major version bump.

**Provider contract** — `ProviderBase` has 2 abstract methods and 3
optional methods with safe defaults. New abstract methods require a major
version bump.

**OTel attributes** — `turnpike.*` custom attributes are covered by
SemVer. `gen_ai.*` attributes follow OTel GenAI Semantic Conventions and
may change upstream.

**Cost model** — pricing values are configuration, not API. They may be
updated in any version. Use `register_pricing()` for custom models.

## Reference Application

The `app/` directory contains a FastAPI application demonstrating
Turnpike in a real HTTP service. It is **not** part of
`pip install turnpike`:

```bash
pip install turnpike[ref]
uvicorn app.main:app --reload
```

## Tests

```bash
OTEL_SDK_DISABLED=true python3 -m pytest tests/ -q
```

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src/turnpike/ --ignore-missing-imports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
