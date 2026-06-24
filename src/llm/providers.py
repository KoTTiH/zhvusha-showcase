"""LLM provider registry — single source of truth for adapters, models,
aliases, and per-token pricing.

Public:
    PROVIDERS — dict of provider name → ProviderSpec
    resolve_model(provider, model) — alias / api_id → ModelSpec | None
    get_pricing(provider, model) — alias / api_id → ModelSpec | None

The router consults this module when bootstrapping. The usage tracker
consults ``get_pricing`` to convert recorded tokens into USD without
hardcoding price tables of its own. Adding a new provider is a single
edit here plus a new adapter file — no changes in router/tracker logic.

Private (gateway-internal): import-linter exposes this module to
``src.llm.router`` and ``src.monitoring`` only. Other capability modules
must depend on the public ``LLMGatewayProtocol`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.llm.anthropic_api import AnthropicAPIAdapter
from src.llm.claude_cli import ClaudeCLIAdapter
from src.llm.codex_cli import CodexCLIAdapter
from src.llm.gemini import GeminiAdapter
from src.llm.openai_compatible import (
    HuggingFaceRouterAdapter,
    MiniMaxAdapter,
    NvidiaNIMAdapter,
)
from src.llm.openrouter import OpenRouterAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.llm.base import BaseLLMAdapter

BillingMode = Literal["per_token", "subscription", "free_tier"]


@dataclass(frozen=True)
class ModelSpec:
    """One model under a provider. ``api_id`` is what the adapter sends
    to the provider's API; ``aliases`` are short forms users put in .env."""

    api_id: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.1
    emoji: str = "⚪"
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProviderSpec:
    """One provider — adapter factory + the models it serves + how it bills."""

    name: str
    billing_mode: BillingMode
    models: dict[str, ModelSpec]
    adapter_factory: Callable[[], BaseLLMAdapter]
    requires_setting: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic_api": ProviderSpec(
        name="anthropic_api",
        billing_mode="per_token",
        models={
            "claude-haiku-4-5-20251001": ModelSpec(
                api_id="claude-haiku-4-5-20251001",
                input_per_mtok=0.80,
                output_per_mtok=4.00,
                emoji="🔵",
                aliases=("haiku",),
            ),
            "claude-sonnet-4-6": ModelSpec(
                api_id="claude-sonnet-4-6",
                input_per_mtok=3.00,
                output_per_mtok=15.00,
                emoji="🟡",
                aliases=("sonnet",),
            ),
            "claude-opus-4-7": ModelSpec(
                api_id="claude-opus-4-7",
                input_per_mtok=15.00,
                output_per_mtok=75.00,
                emoji="🟣",
                aliases=("opus",),
            ),
        },
        adapter_factory=AnthropicAPIAdapter,
        requires_setting="anthropic_api_key",
    ),
    "claude_cli": ProviderSpec(
        name="claude_cli",
        billing_mode="subscription",
        models={
            "haiku": ModelSpec(api_id="haiku", emoji="🔵"),
            "sonnet": ModelSpec(api_id="sonnet", emoji="🟡"),
            "opus": ModelSpec(api_id="opus", emoji="🟣"),
        },
        adapter_factory=ClaudeCLIAdapter,
        requires_setting=None,
    ),
    "codex_cli": ProviderSpec(
        name="codex_cli",
        billing_mode="subscription",
        models={
            "default": ModelSpec(api_id="default", emoji="⚫", aliases=("codex",)),
            "gpt-5.5": ModelSpec(
                api_id="gpt-5.5",
                emoji="⚫",
                aliases=("best", "frontier"),
            ),
        },
        adapter_factory=CodexCLIAdapter,
        requires_setting=None,
    ),
    "gemini": ProviderSpec(
        name="gemini",
        billing_mode="free_tier",
        models={
            "gemini-2.5-flash-lite": ModelSpec(
                api_id="gemini-2.5-flash-lite", emoji="🟢"
            ),
            "gemini-2.5-flash": ModelSpec(api_id="gemini-2.5-flash", emoji="🟢"),
        },
        adapter_factory=GeminiAdapter,
        requires_setting="google_api_key",
    ),
    # OpenRouter aggregates many vendors (DeepSeek, Llama, Qwen, Mistral,
    # GPT, Gemini-via-router, ...) under a single OpenAI-compatible API.
    # Models are addressed as ``vendor/model``; api_id is passed through
    # unchanged. Pricing here matches OpenRouter's listed rates as of
    # April 2026 and may need a refresh — verify at openrouter.ai/models.
    "openrouter": ProviderSpec(
        name="openrouter",
        billing_mode="per_token",
        models={
            # DeepSeek V4 (released 2026-04-23/24). Pricing pulled live
            # from openrouter.ai/api/v1/models on 2026-04-25 — refresh
            # when the rates drift.
            "deepseek/deepseek-v4-pro": ModelSpec(
                api_id="deepseek/deepseek-v4-pro",
                input_per_mtok=1.74,
                output_per_mtok=3.48,
                emoji="🔷",
                aliases=("deepseek-v4-pro", "v4-pro"),
            ),
            "deepseek/deepseek-v4-flash": ModelSpec(
                api_id="deepseek/deepseek-v4-flash",
                input_per_mtok=0.14,
                output_per_mtok=0.28,
                emoji="🔷",
                aliases=("deepseek-v4-flash", "v4-flash"),
            ),
            "deepseek/deepseek-chat": ModelSpec(
                api_id="deepseek/deepseek-chat",
                input_per_mtok=0.27,
                output_per_mtok=1.10,
                emoji="🔷",
                aliases=("deepseek", "deepseek-v3"),
            ),
            "deepseek/deepseek-r1": ModelSpec(
                api_id="deepseek/deepseek-r1",
                input_per_mtok=0.55,
                output_per_mtok=2.19,
                emoji="🔷",
                aliases=("deepseek-r1",),
            ),
        },
        adapter_factory=OpenRouterAdapter,
        requires_setting="openrouter_api_key",
    ),
    "nvidia_nim": ProviderSpec(
        name="nvidia_nim",
        billing_mode="per_token",
        models={
            "nvidia/llama-3.1-nemotron-ultra-253b-v1": ModelSpec(
                api_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                emoji="🟩",
                aliases=("nim-default", "nemotron-ultra"),
            ),
        },
        adapter_factory=NvidiaNIMAdapter,
        requires_setting="nvidia_nim_api_key",
    ),
    "huggingface": ProviderSpec(
        name="huggingface",
        billing_mode="per_token",
        models={
            "deepseek-ai/DeepSeek-R1:fastest": ModelSpec(
                api_id="deepseek-ai/DeepSeek-R1:fastest",
                emoji="🤗",
                aliases=("hf-default", "hf-deepseek-r1"),
            ),
        },
        adapter_factory=HuggingFaceRouterAdapter,
        requires_setting="huggingface_api_key",
    ),
    "minimax": ProviderSpec(
        name="minimax",
        billing_mode="per_token",
        models={
            "MiniMax-M2.7": ModelSpec(
                api_id="MiniMax-M2.7",
                emoji="🟧",
                aliases=("m2-7", "minimax-m2-7"),
            ),
            "MiniMax-M2.7-highspeed": ModelSpec(
                api_id="MiniMax-M2.7-highspeed",
                emoji="🟧",
                aliases=("m2-7-fast",),
            ),
        },
        adapter_factory=MiniMaxAdapter,
        requires_setting="minimax_api_key",
    ),
}


def resolve_model(provider: str, model: str) -> ModelSpec | None:
    """Return the canonical ModelSpec for ``model`` under ``provider``,
    accepting either an alias (``haiku``) or the api_id
    (``claude-haiku-4-5-20251001``). Returns ``None`` for unknown providers
    or unknown models — caller decides what an unknown model means
    (router rejects, tracker treats as zero-cost passthrough)."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        return None
    if model in spec.models:
        return spec.models[model]
    for ms in spec.models.values():
        if model in ms.aliases:
            return ms
    return None


def get_pricing(provider: str, model: str) -> ModelSpec | None:
    """Public lookup for the usage tracker. Currently identical to
    :func:`resolve_model`, kept as a separate name so the read-only
    intent is explicit at the call site."""
    return resolve_model(provider, model)
