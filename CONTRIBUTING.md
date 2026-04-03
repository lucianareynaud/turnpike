# Contributing to Turnpike

## Development Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the library in editable mode with all provider SDKs, the reference app dependencies, and dev tooling (pytest, ruff, mypy).

## Running Tests

```bash
OTEL_SDK_DISABLED=true python3 -m pytest tests/ -q
```

Linting and formatting:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
```

Type checking:

```bash
python3 -m mypy src/turnpike/ --ignore-missing-imports
```

All three gates must pass before merging. CI runs them automatically via GitHub Actions.

## Adding a New Provider

1. Create a class that subclasses `ProviderBase` in `src/turnpike/gateway/provider.py` (or a new file under `src/turnpike/gateway/providers/` if the module grows):

```python
from turnpike.gateway.provider import ProviderBase, ProviderResponse

class BedrockProvider(ProviderBase):
    @property
    def provider_name(self) -> str:
        return "bedrock"

    async def complete(self, model, messages, **kwargs) -> ProviderResponse:
        # Provider-specific API call
        return ProviderResponse(content="...", model=model, usage={...})

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, SomeTransientError)

    def categorize_error(self, error: Exception) -> str:
        return "rate_limit" if isinstance(error, RateLimitError) else "unknown"
```

2. Register the provider so the gateway can resolve it:

```python
from turnpike import register_provider
register_provider("bedrock", BedrockProvider())
```

3. Add pricing data to `src/turnpike/gateway/cost_model.py` so `estimate_cost()` returns accurate values for the new provider's models.

4. If the provider SDK is an optional dependency, add it to `pyproject.toml` under `[project.optional-dependencies]` following the existing pattern (e.g. `bedrock = ["boto3>=1.34"]`).

5. Write tests. At minimum: a unit test for `complete()` with a mocked SDK client, and a test verifying `estimate_cost()` returns non-zero for the new models.

## Adding a New Route Policy

Route policies map complexity tiers to specific models. To register a custom policy:

```python
from turnpike import register_route_policy, RoutePolicy

register_route_policy(
    "/classify",
    RoutePolicy(
        max_output_tokens=500,
        retry_attempts=2,
        cache_enabled=False,
        model_for_tier={"cheap": "gpt-4o-mini", "expensive": "gpt-4o"},
        provider_name="openai",
    ),
)
```

Policies are resolved at `call_llm()` time via the `route_name` parameter. If no policy is registered for a route, the gateway raises `ValueError`.

## Code Style

Turnpike uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration lives in `pyproject.toml`:

- Line length: 100
- Lint rules: `E`, `F`, `I`, `UP`, `W`, `B`, `C4`
- Quote style: double
- Indent style: spaces

Run `ruff format .` to auto-format before committing.

## Pull Request Expectations

- One logical change per PR. Separate refactors from feature work.
- All CI gates must pass (ruff, mypy, pytest).
- New public API surface must be added to `turnpike.__all__` in `src/turnpike/__init__.py`.
- New features require tests. Bug fixes require a regression test.
- Envelope schema changes must increment `schema_version` and document the change in `CHANGELOG.md`.
- Provider SDK dependencies must remain optional extras, never hard dependencies.
- Keep commit messages concise. Describe the "why", not the "what".

## Project Structure

```
src/turnpike/           ← pip-installable library
  __init__.py           ← public API surface (__all__)
  envelope.py           ← LLMRequestEnvelope dataclass
  context.py            ← LLMRequestContext
  gateway/
    client.py           ← call_llm() — the gateway choke point
    provider.py         ← ProviderBase, built-in providers
    telemetry.py        ← emit_event() — OTel + JSONL dual-write
    cost_model.py       ← estimate_cost(), pricing data
    policies.py         ← RoutePolicy, register_route_policy
    otel_setup.py       ← setup_otel(), shutdown_otel()
    semconv.py          ← OTel attribute constants
app/                    ← reference FastAPI app (not pip-installed)
evals/                  ← eval harness
tests/                  ← test suite
```
