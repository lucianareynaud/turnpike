[![PyPI](https://img.shields.io/pypi/v/turnpike?v=1)](https://pypi.org/project/turnpike/)
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

Distributed tracing was built for request-reply microservices. LLM
systems break that model: costs compound across retries and fallbacks,
and a single "request" may span multiple providers.

Standard tracing gives you call visibility. Turnpike adds structured
cost attribution — who spent what, on which model, under which policy.

| Concern | Tracing alone | Turnpike |
|---|---|---|
| Token counts | Manual span attributes | `tokens_in` / `tokens_out` / `tokens_total` |
| Cost attribution | Not modeled | `estimated_cost_usd` + `cost_source` provenance |
| Multi-tenant billing | Custom baggage | `tenant_id` + `caller_id` + `budget_namespace` |
| Policy enforcement | Out of scope | `policy_decision` + `redaction_applied` |
| Retry / fallback | Span count heuristics | `retry_count` + `fallback_triggered` + `circuit_state` |
| Streaming latency | No convention | `time_to_first_token_ms` + `finish_reason` |

**What Turnpike is not.** It is not a proxy, not a dashboard, and not a
prompt management tool. It is a typed envelope and telemetry layer that
sits between your LLM calls and your observability backend.

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
envelope.

Gateway examples use OpenAI. Requires `pip install turnpike[openai]`
and a valid `OPENAI_API_KEY`.

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

Pass a full conversation history using `messages` instead of `prompt`.
Assumes the `/summarize` route policy from the previous example is
already registered.

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
    print(result.text)

asyncio.run(main())
```

### Streaming

Stream responses token-by-token. The `GatewayStream` yields `StreamChunk`
objects; after iteration, `.result` holds the full `GatewayResult` with
envelope, cost, and TTFT. Assumes the `/summarize` route policy is
registered.

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

`LLMRequestEnvelope` is a frozen dataclass. Four fields are required;
the rest default to `None` or safe enum values.

**Required:** `schema_version`, `request_id`, `tenant_id`, `route`

**Optional fields by semantic group:**

- **Identity / context** — `caller_id`, `session_id`, `task_id`, `use_case`, `trace_id`, `span_id`
- **Model selection** — `provider_requested`, `provider_selected`, `model_requested`, `model_selected`, `model_tier`, `routing_decision`, `routing_reason`
- **Economics** — `tokens_in`, `tokens_out`, `tokens_total`, `estimated_cost_usd`, `cost_source` (default `DEGRADED_UNKNOWN`)
- **Reliability** — `latency_ms`, `time_to_first_token_ms`, `status` (default `OK`), `error_type`, `retry_count`, `fallback_triggered`, `fallback_reason`, `circuit_state`, `streaming`, `finish_reason`
- **Governance** — `policy_input_class`, `policy_decision`, `policy_mode`, `redaction_applied`, `pii_detected`
- **Cache / eval** — `cache_eligible`, `cache_strategy`, `cache_hit`, `cache_key_fingerprint`, `cache_key_algorithm`, `cache_lookup_confidence`, `eval_hooks`, `audit_tags`

Full type annotations are in [`envelope.py`](src/turnpike/envelope.py).
New optional fields may appear in minor versions. Consumers should
tolerate unknown keys. Removing or renaming a field requires a major
version bump.

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
internal helpers are importable from their submodules but are not part
of the top-level contract.

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

This package follows [Semantic Versioning](https://semver.org/).

- **Public API** — all names in `turnpike.__all__` are covered by SemVer. Internal module paths may change in minor versions.
- **Envelope schema** — new optional fields may appear in minor versions. Removing or renaming fields requires a major version bump.
- **Provider contract** — `provider_name` and `complete()` are abstract and stable. New abstract methods require a major version bump.
- **OTel attributes** — `turnpike.*` attributes are covered by SemVer. `gen_ai.*` attributes follow upstream OTel conventions.
- **Cost model** — pricing values are configuration, not API, and may change in any version.

## Reference Application

The `app/` directory contains a FastAPI application demonstrating
Turnpike in a real HTTP service. It is **not** part of
`pip install turnpike`. To run it, clone the repo and install the
reference dependencies:

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
