"""Public contract for the LLM Gateway capability module (v4).

All other modules import types, errors, and the protocol from THIS FILE ONLY.
Concrete adapters (Codex CLI, Anthropic SDK, Google GenAI, and legacy Claude
CLI) are hidden behind
``LLMGatewayProtocol``. ``import-linter`` enforces this isolation via the
``llm_gateway_isolation`` rule in ``.importlinter``.

Side effects the gateway performs
---------------------------------
- Calls external LLM APIs.
- May spawn subscription-backed CLI subprocesses (Codex by default; Claude
  only when explicitly configured as a legacy LLM provider).
- Tracks token usage and cost via ``UsageTracker`` when one is attached on
  the concrete router.

Errors
------
- ``LLMError`` — base exception raised by any adapter on failure.
- ``RateLimitError`` — provider rate limit (reserved for future use).
- ``BudgetExceededError`` — SafetyGuard cap (reserved).
- ``AuthenticationError`` — OAuth / API key invalid (reserved).
- ``ProviderUnavailableError`` — unrecoverable provider error (reserved).

The reserved error classes are defined so future phases (SafetyGuard,
retry/backoff loop) can raise them without changing the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from src.core.config import ReasoningEffort, Tier

__all__ = [
    "DEFAULT_VISION_PROMPT",
    "AuthenticationError",
    "BudgetExceededError",
    "LLMError",
    "LLMGatewayProtocol",
    "LLMImageRequest",
    "LLMImageResponse",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMToolRequest",
    "LLMToolResponse",
    "LLMUsage",
    "LLMVisionRequest",
    "ProviderUnavailableError",
    "RateLimitError",
    "Tier",
    "ToolDefinition",
]


DEFAULT_VISION_PROMPT = "Опиши что ты видишь на изображении(ях). Детально."


# === Data classes ===


@dataclass(frozen=True)
class LLMMessage:
    """Chat message — reserved for future multi-turn API.

    Currently ``LLMRequest`` takes a raw ``prompt: str`` because every existing
    caller builds its own prompt string. ``LLMMessage`` is kept in the contract
    so future phases can introduce multi-turn without re-defining the type.
    """

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ToolDefinition:
    """Tool definition for LLM tool_use. Moved from ``src.llm.base`` in phase 2."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class LLMUsage:
    """Token usage and caching metadata for a single LLM call.

    Adapters that do not report usage (subscription CLI, Gemini free tier) return
    a zero-valued instance. Callers can rely on the shape being present.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class LLMRequest:
    """Single-shot text generation request."""

    prompt: str
    system: str = ""
    tier: Tier = "worker"
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
    caller: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """Single-shot text generation response."""

    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass(frozen=True)
class LLMToolRequest:
    """Tool-use generation request.

    ``messages`` stays provider-native (Anthropic shape
    ``[{"role": ..., "content": ...}]`` with possible nested tool_use /
    tool_result blocks) so callers can thread an agentic loop without a lossy
    translation layer.
    """

    messages: list[dict[str, Any]]
    tools: list[ToolDefinition]
    system: str = ""
    tier: Tier = "analyst"
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
    caller: str = ""


@dataclass(frozen=True)
class LLMToolResponse:
    """Tool-use generation response.

    ``content_blocks`` are provider-native (Anthropic ``TextBlock`` /
    ``ToolUseBlock``). The caller manages the agentic loop — parses blocks,
    runs tools, builds the next ``LLMToolRequest``.
    """

    content_blocks: list[Any]
    stop_reason: str | None
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass(frozen=True)
class LLMVisionRequest:
    """Vision request — describe one or more images."""

    images: list[bytes]
    prompt: str = DEFAULT_VISION_PROMPT
    caller: str = ""


@dataclass(frozen=True)
class LLMImageRequest:
    """Generated-image request routed through a configured provider."""

    prompt: str
    model: str | None = None
    size: str | None = None
    caller: str = ""


@dataclass(frozen=True)
class LLMImageResponse:
    """Generated-image response with bytes ready for artifact storage."""

    image: bytes
    model: str
    mime_type: str = "image/png"
    revised_prompt: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)


# === Errors ===


class LLMError(Exception):
    """Base exception raised by any adapter on failure (network, auth, empty)."""


class RateLimitError(LLMError):
    """Provider rate limit hit. Retry after backoff. Reserved for future use."""


class BudgetExceededError(LLMError):
    """Local budget cap exceeded. Reserved for SafetyGuard integration."""


class AuthenticationError(LLMError):
    """OAuth token expired or API key invalid. Reserved for future use."""


class ProviderUnavailableError(LLMError):
    """Provider returned an unrecoverable error. Reserved for future use."""


# === Protocol ===


@runtime_checkable
class LLMGatewayProtocol(Protocol):
    """Public contract for the LLM Gateway capability module.

    The concrete implementation is ``LLMRouter`` in ``src.llm.router``.
    Clients depend on this protocol, not the concrete class, so the
    implementation can be swapped for tests or future providers.
    """

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Single-shot text generation.

        Resolves ``request.tier`` to a concrete model via the router config,
        dispatches to the appropriate adapter, and tracks usage when a tracker
        is attached to the router.

        Raises:
            LLMError: adapter failure (network, auth, empty response, ...).
        """
        ...

    async def generate_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        """Tool-use generation. Returns provider-native content blocks.

        Raises:
            LLMError: adapter failure.
            NotImplementedError: the routed adapter does not support tool_use.
        """
        ...

    async def describe_images(self, request: LLMVisionRequest) -> LLMResponse:
        """Vision: describe one or more images.

        Returns an ``LLMResponse`` whose ``text`` holds the description. If no
        vision adapter is configured on the router, returns a fallback message
        rather than raising — matches the existing ``LLMRouter`` behavior.
        """
        ...

    async def generate_image(self, request: LLMImageRequest) -> LLMImageResponse:
        """Generate one image through the optional configured image provider.

        Raises:
            ProviderUnavailableError: image generation is disabled/unconfigured.
            LLMError: provider failure.
        """
        ...
