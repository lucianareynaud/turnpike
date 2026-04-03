# Changelog

All notable changes to Turnpike will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-03

Initial public release. Intentionally narrow core focused on typed LLM
cost attribution with OpenTelemetry instrumentation.

### Added

- `LLMRequestEnvelope` — frozen, versioned dataclass for LLM request lifecycle
- `LLMRequestContext` — structured attribution context (tenant, caller, session, task)
- Gateway entry points: `call_llm()` and `call_llm_stream()` with keyword-only signatures
- `ProviderBase` ABC with forward-compatible evolution contract
- Built-in `OpenAIProvider` and `AnthropicProvider` as optional extras
- `RoutePolicy` and `register_route_policy()` for per-route model tier mapping
- `register_provider()` for runtime provider registration
- `register_pricing()` for custom model pricing
- `emit_event()` for lifecycle event emission to JSONL telemetry
- OpenTelemetry instrumentation: CLIENT spans with GenAI semantic conventions
- 4 metric instruments: token usage, operation duration, estimated cost, request count
- Deterministic cost model with pricing for OpenAI and Anthropic models
- `src/` layout with PEP 561 `py.typed` marker
- Streaming support via `GatewayStream` and `StreamChunk` with TTFT measurement

### Provider SDK dependencies

Provider SDKs are optional extras, not hard dependencies:

```bash
pip install turnpike[openai]      # OpenAI only
pip install turnpike[anthropic]   # Anthropic only
pip install turnpike[all]         # Both providers
```

[Unreleased]: https://github.com/lucianareynaud/turnpike/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lucianareynaud/turnpike/releases/tag/v0.1.0
