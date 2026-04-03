"""Route policy configuration for the gateway.

Policies map route names to model tier configurations. All policies are
registered at runtime via ``register_route_policy()``. The registry starts
empty; callers must register policies before calling ``call_llm()``.

Each RoutePolicy includes a ``provider_name`` field that determines which
registered provider (see ``gateway/provider.py``) handles the LLM call.
The default is ``"openai"``.
"""

from dataclasses import dataclass

__all__ = [
    "RoutePolicy",
    "get_route_policy",
    "get_model_for_tier",
    "register_route_policy",
    "clear_route_policies",
]

ModelTier = str


@dataclass(frozen=True)
class RoutePolicy:
    """Policy configuration for a gateway-backed route.

    Attributes:
        max_output_tokens: Token cap for provider response.
        retry_attempts: Number of additional attempts after first failure.
        cache_enabled: Whether semantic cache is enabled for this route.
        model_for_tier: Maps logical tier to concrete model name.
        provider_name: Registered provider to use (default: "openai").
    """

    max_output_tokens: int
    retry_attempts: int
    cache_enabled: bool
    model_for_tier: dict[ModelTier, str]
    provider_name: str = "openai"


_ROUTE_POLICIES: dict[str, RoutePolicy] = {}


def get_route_policy(route_name: str) -> RoutePolicy:
    """Return the policy configuration for a gateway-backed route.

    Args:
        route_name: Route path (e.g. ``"/summarize"``).

    Returns:
        The configured RoutePolicy for the route.

    Raises:
        ValueError: If the route has no gateway policy.
    """
    if route_name not in _ROUTE_POLICIES:
        if not _ROUTE_POLICIES:
            raise ValueError(
                f"No gateway policy defined for route: {route_name}. "
                "No policies registered. Call register_route_policy() first."
            )
        available_routes = ", ".join(sorted(_ROUTE_POLICIES.keys()))
        raise ValueError(
            f"No gateway policy defined for route: {route_name}. "
            f"Registered routes: {available_routes}"
        )

    return _ROUTE_POLICIES[route_name]


def register_route_policy(route_name: str, policy: RoutePolicy) -> None:
    """Register or replace a route policy at runtime.

    Args:
        route_name: Route path such as ``/my-custom-route``.
        policy: RoutePolicy instance to associate with the route.
    """
    _ROUTE_POLICIES[route_name] = policy


def clear_route_policies() -> None:
    """Remove all route policies (useful in tests)."""
    _ROUTE_POLICIES.clear()


def get_model_for_tier(route_name: str, tier: ModelTier) -> str:
    """Resolve the concrete model name for a logical tier on a route.

    Args:
        route_name: Route path (e.g. ``"/summarize"``).
        tier: Logical model tier.

    Returns:
        Concrete model name configured for that route and tier.

    Raises:
        ValueError: If the tier is not configured for the route.
    """
    policy = get_route_policy(route_name)

    if tier not in policy.model_for_tier:
        available_tiers = ", ".join(sorted(policy.model_for_tier.keys()))
        raise ValueError(
            f"Tier '{tier}' not configured for route {route_name}. "
            f"Available tiers: {available_tiers}"
        )

    return policy.model_for_tier[tier]
