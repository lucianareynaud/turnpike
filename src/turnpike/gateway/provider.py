"""Provider abstraction for the Turnpike gateway.

ARCHITECTURAL ROLE
──────────────────
This module decouples the gateway from any specific LLM provider SDK.
The gateway (client.py) handles retry, telemetry, cost estimation, OTel spans,
and envelope construction. The provider handles ONLY the API call and response
normalization.

This separation means adding a new provider requires:
1. Subclass ProviderBase (one class, two required methods).
2. Register via register_provider() or auto-registration.
3. Add pricing to cost_model.py.
4. No changes to client.py, telemetry.py, or any route handler.

BACKWARD COMPATIBILITY CONTRACT
─────────────────────────────────
ProviderBase uses an ABC with abstract + default methods, not a Protocol.
This is a deliberate design decision for forward-compatible evolution:

  ABSTRACT (must override — breaking to add new ones in minor versions):
    - provider_name     — stable since v0.1.0
    - complete()        — stable since v0.1.0

  DEFAULT (safe to add new ones — existing subclasses inherit the default):
    - is_retryable()    — default returns False (no retry)
    - categorize_error() — default returns "unknown"
    - stream()          — default falls back to complete() and yields one event

If a future version adds estimate_tokens(), it will be a default method.
Existing third-party providers will not break.

Adding a new abstract method is a MAJOR version bump (breaking change).

BUILT-IN PROVIDERS
───────────────────
  OpenAIProvider    — requires ``pip install turnpike[openai]``
  AnthropicProvider — requires ``pip install turnpike[anthropic]``

Both are optional extras. The core package (envelope, telemetry, gateway)
installs with zero provider SDKs.

AUTO-REGISTRATION
──────────────────
At module import time, each built-in provider checks whether its SDK is
importable. If yes, an instance is registered in the module-level registry.
If not, the provider is silently skipped.

TESTING
────────
Tests that call ``call_llm()`` should patch ``gateway.client.get_provider``
to return a mock provider:

    mock_provider = Mock(spec=ProviderBase)
    mock_provider.provider_name = "mock"
    mock_provider.complete = AsyncMock(return_value=ProviderResponse(...))

    with patch("gateway.client.get_provider", return_value=mock_provider):
        result = await call_llm(...)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

__all__ = [
    "ProviderBase",
    "ProviderResponse",
    "ProviderStreamEvent",
    "OpenAIProvider",
    "AnthropicProvider",
    "register_provider",
    "get_provider",
    "available_providers",
]

Message = dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response from any LLM provider.

    Frozen dataclass — immutable after creation. The gateway consumes this
    directly for telemetry, cost estimation, and envelope construction.
    """

    text: str
    tokens_in: int
    tokens_out: int
    finish_reason: str | None = None
    response_model: str | None = None


@dataclass(frozen=True)
class ProviderStreamEvent:
    """A single event from a streaming provider response.

    During streaming, the gateway receives a sequence of these events.
    Most events carry a text ``delta``. The final event carries usage
    metadata (tokens_in, tokens_out, finish_reason) needed for envelope
    and telemetry construction.
    """

    delta: str = ""
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    response_model: str | None = None


class ProviderBase(ABC):
    """Base class for LLM provider adapters.

    The gateway calls ``complete()`` and handles everything else:
    retry with exponential backoff, OTel span lifecycle, cost estimation,
    envelope construction, metric emission, JSONL artifact.

    Subclass contract:
      REQUIRED (abstract — must override):
        provider_name   — stable identifier for telemetry
        complete()      — make the API call and normalize the response

      OPTIONAL (have safe defaults — override for provider-specific behavior):
        is_retryable()    — default: return False (never retry)
        categorize_error() — default: return "unknown"
        stream()          — default: falls back to complete()

    This design ensures new optional methods can be added in minor versions
    without breaking existing subclasses.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable identifier for telemetry attributes.

        This string appears in OTel span attributes, JSONL telemetry events,
        and the LLMRequestEnvelope.provider_selected field. Must be lowercase,
        stable across releases, and never contain version info.

        Examples: ``"openai"``, ``"anthropic"``, ``"google"``, ``"bedrock"``
        """
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        """Execute one completion call and return normalized response.

        Called inside the gateway's retry loop. Must make exactly one API
        call and either return a ``ProviderResponse`` or raise an exception.
        The gateway handles retry decisions via ``is_retryable()``.

        Args:
            messages: List of message dicts (e.g. ``[{"role": "user", "content": "..."}]``).
            model: Concrete model name (e.g. ``"gpt-4o-mini"``).
            max_output_tokens: Token cap for the response.

        Returns:
            ProviderResponse with normalized text and token counts.

        Raises:
            Any exception from the provider SDK. The gateway will call
            ``is_retryable()`` and ``categorize_error()`` on it.
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Stream a completion, yielding events as they arrive.

        The default implementation falls back to ``complete()`` and yields
        a single event containing the full response. Override in providers
        that support native streaming for real token-by-token delivery.

        The final event MUST include ``tokens_in``, ``tokens_out``, and
        ``finish_reason`` so the gateway can build the envelope.

        Args:
            messages: List of message dicts.
            model: Concrete model name.
            max_output_tokens: Token cap for the response.

        Yields:
            ProviderStreamEvent instances.
        """
        response = await self.complete(messages, model, max_output_tokens)
        yield ProviderStreamEvent(
            delta=response.text,
            finish_reason=response.finish_reason,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            response_model=response.response_model,
        )

    def is_retryable(self, error: Exception) -> bool:
        """Return True if the error should trigger a retry attempt.

        Override with provider-specific retry logic. Default: returns False
        (never retry). This is the safe default because retrying an unknown
        error can cause duplicate requests.
        """
        return False

    def categorize_error(self, error: Exception) -> str:
        """Map an exception to a stable telemetry error category string.

        Override with provider-specific error classification. Default:
        returns "unknown".

        Use a consistent taxonomy across providers:
          ``auth_error``       — 401/403
          ``rate_limit``       — 429
          ``timeout``          — network timeout
          ``transient_error``  — network failure or 5xx
          ``invalid_request``  — 400/404/422
          ``unknown``          — none of the above
        """
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Provider Registry
# ─────────────────────────────────────────────────────────────────────────────

_registry: dict[str, ProviderBase] = {}


def register_provider(name: str, provider: ProviderBase) -> None:
    """Register a provider implementation by name."""
    _registry[name] = provider


def get_provider(name: str, auto_register: bool = True) -> ProviderBase:
    """Resolve a registered provider by name with lazy auto-registration.

    On first access for a built-in provider, attempts to import the SDK
    and register the provider automatically.

    Raises:
        ValueError: If the provider is not registered and cannot be auto-registered.
    """
    if name not in _registry and auto_register:
        _try_auto_register(name)
    if name not in _registry:
        installed = list(_registry.keys()) or ["none"]
        raise ValueError(
            f"Provider '{name}' is not registered. "
            f"Installed providers: {', '.join(installed)}. "
            f"Install the SDK: pip install turnpike[{name}]"
        )
    return _registry[name]


def available_providers() -> list[str]:
    """Return names of all currently registered providers."""
    return list(_registry.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Built-in: OpenAI
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIProvider(ProviderBase):
    """OpenAI provider via the Responses API.

    Requires: ``pip install turnpike[openai]``
    Env var: ``OPENAI_API_KEY``
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize and return the cached AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self._client = AsyncOpenAI(api_key=api_key)

        return self._client

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        client = self._get_client()
        response = await client.responses.create(
            model=model,
            input=messages,  # type: ignore[arg-type]
            max_output_tokens=max_output_tokens,
        )

        text = response.output_text or ""
        usage = response.usage
        tokens_in = usage.input_tokens if usage else 0
        tokens_out = usage.output_tokens if usage else 0

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            finish_reason=response.status,
            response_model=response.model,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> AsyncIterator[ProviderStreamEvent]:
        client = self._get_client()
        stream = await client.responses.create(
            model=model,
            input=messages,  # type: ignore[arg-type]
            max_output_tokens=max_output_tokens,
            stream=True,
        )

        async for event in stream:  # type: ignore[union-attr]
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                yield ProviderStreamEvent(delta=getattr(event, "delta", ""))
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    usage = resp.usage
                    yield ProviderStreamEvent(
                        delta="",
                        finish_reason=resp.status,
                        tokens_in=usage.input_tokens if usage else 0,
                        tokens_out=usage.output_tokens if usage else 0,
                        response_model=resp.model,
                    )

    def is_retryable(self, error: Exception) -> bool:
        import openai

        return isinstance(
            error,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        )

    def categorize_error(self, error: Exception) -> str:
        import openai

        if isinstance(error, openai.AuthenticationError):
            return "auth_error"
        if isinstance(error, openai.PermissionDeniedError):
            return "auth_error"
        if isinstance(error, openai.RateLimitError):
            return "rate_limit"
        if isinstance(error, openai.APITimeoutError):
            return "timeout"
        if isinstance(error, openai.APIConnectionError):
            return "transient_error"
        if isinstance(error, openai.InternalServerError):
            return "transient_error"
        if isinstance(
            error,
            (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError),
        ):
            return "invalid_request"
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Built-in: Anthropic
# ─────────────────────────────────────────────────────────────────────────────


class AnthropicProvider(ProviderBase):
    """Anthropic provider via the Messages API.

    Requires: ``pip install turnpike[anthropic]``
    Env var: ``ANTHROPIC_API_KEY``
    """

    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> AsyncAnthropic:
        """Lazy-initialize and return the cached AsyncAnthropic client."""
        if self._client is None:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            self._client = anthropic.AsyncAnthropic(api_key=api_key)

        return self._client

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> ProviderResponse:
        client = self._get_client()
        response = await client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            messages=messages,  # type: ignore[arg-type]
        )

        text = getattr(response.content[0], "text", "") if response.content else ""
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            finish_reason=response.stop_reason,
            response_model=response.model,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str,
        max_output_tokens: int,
    ) -> AsyncIterator[ProviderStreamEvent]:
        client = self._get_client()
        async with client.messages.stream(
            model=model,
            max_tokens=max_output_tokens,
            messages=messages,  # type: ignore[arg-type]
        ) as stream:
            async for text in stream.text_stream:
                yield ProviderStreamEvent(delta=text)

            final = await stream.get_final_message()
            yield ProviderStreamEvent(
                delta="",
                finish_reason=final.stop_reason,
                tokens_in=final.usage.input_tokens,
                tokens_out=final.usage.output_tokens,
                response_model=final.model,
            )

    def is_retryable(self, error: Exception) -> bool:
        try:
            import anthropic
        except ImportError:
            return False

        return isinstance(
            error,
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )

    def categorize_error(self, error: Exception) -> str:
        try:
            import anthropic
        except ImportError:
            return "unknown"

        if isinstance(error, anthropic.AuthenticationError):
            return "auth_error"
        if isinstance(error, anthropic.PermissionDeniedError):
            return "auth_error"
        if isinstance(error, anthropic.RateLimitError):
            return "rate_limit"
        if isinstance(error, anthropic.APITimeoutError):
            return "timeout"
        if isinstance(error, anthropic.APIConnectionError):
            return "transient_error"
        if isinstance(error, anthropic.InternalServerError):
            return "transient_error"
        if isinstance(error, (anthropic.BadRequestError, anthropic.NotFoundError)):
            return "invalid_request"
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy auto-registration (opt-in on first access via get_provider)
# ─────────────────────────────────────────────────────────────────────────────

_AUTO_PROVIDERS: dict[str, type[ProviderBase]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def _try_auto_register(name: str) -> None:
    """Attempt to import the SDK for *name* and register if available."""
    if name not in _AUTO_PROVIDERS:
        return
    provider_cls = _AUTO_PROVIDERS[name]
    try:
        __import__(name)
        register_provider(name, provider_cls())
    except ImportError:
        pass
