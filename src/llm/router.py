"""LLM Router — concrete implementation of ``LLMGatewayProtocol``.

Routes typed requests by tier to the appropriate adapter, resolves model
names, and records usage via the optional ``UsageTracker``. External modules
should depend on ``LLMGatewayProtocol`` (from ``src.llm.protocols``) rather
than this concrete class.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog

from src.core.config import ReasoningEffort, Tier, get_settings
from src.llm.protocols import (
    DEFAULT_VISION_PROMPT,
    LLMError,
    LLMGatewayProtocol,
    LLMImageRequest,
    LLMImageResponse,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
    LLMVisionRequest,
    ProviderUnavailableError,
)
from src.llm.providers import PROVIDERS, resolve_model

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.llm.base import BaseLLMAdapter
    from src.llm.gemini import GeminiAdapter
    from src.monitoring.usage_tracker import UsageTracker


class ImageGeneratorProtocol(Protocol):
    async def generate_image(self, request: LLMImageRequest) -> LLMImageResponse: ...


logger = structlog.get_logger()


TIER_KEYS: dict[Tier, tuple[str, str]] = {
    "worker": ("worker_provider", "worker_model"),
    "analyst": ("analyst_provider", "analyst_model"),
    "strategist": ("strategist_provider", "strategist_model"),
}
TIER_REASONING_KEYS: dict[Tier, str] = {
    "worker": "worker_reasoning_effort",
    "analyst": "analyst_reasoning_effort",
    "strategist": "strategist_reasoning_effort",
}


class LLMRouter(LLMGatewayProtocol):
    """Concrete LLM Gateway. Routes requests by tier to the right adapter.

    All text tiers (worker/analyst/strategist) dispatch to a text adapter
    from ``self._adapters``. Vision (image description) routes to an
    optional ``GeminiAdapter`` if configured. ``UsageTracker`` is attached
    post-construction via ``set_usage_tracker`` by the bot bootstrap code.

    Explicit ``LLMGatewayProtocol`` inheritance lets mypy verify that every
    protocol method is implemented with a compatible signature.
    """

    def __init__(
        self,
        adapters: dict[Tier, BaseLLMAdapter],
        models: dict[Tier, str],
        *,
        reasoning_efforts: dict[Tier, ReasoningEffort] | None = None,
        providers_by_tier: dict[Tier, str] | None = None,
        vision_adapter: GeminiAdapter | None = None,
        vision_provider: str = "gemini",
        image_generator: ImageGeneratorProtocol | None = None,
        image_provider: str = "",
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self._adapters = adapters
        self._models = models
        self._reasoning_efforts: dict[Tier, ReasoningEffort] = reasoning_efforts or {}
        self._providers_by_tier: dict[Tier, str] = providers_by_tier or {}
        self._vision = vision_adapter
        self._vision_provider = vision_provider
        self._image_generator = image_generator
        self._image_provider = image_provider
        self._usage = usage_tracker
        # ``generate_oneoff`` builds adapters on demand for providers that
        # are not wired into any tier (e.g. ``openrouter`` for /compare).
        # Cached so repeated A/B calls hit one HTTP client per provider.
        self._oneoff_adapters: dict[str, BaseLLMAdapter] = {}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        adapter = self._adapters[request.tier]
        resolved_model = request.model or self._models.get(
            request.tier, adapter.default_model
        )
        resolved_effort = request.reasoning_effort or self._reasoning_efforts.get(
            request.tier
        )
        adapter_request = replace(
            request,
            model=resolved_model,
            reasoning_effort=resolved_effort,
        )
        logger.debug(
            "llm_route",
            tier=request.tier,
            adapter=adapter.name,
            model=resolved_model,
            reasoning_effort=resolved_effort,
            caller=request.caller,
        )
        response = await adapter.generate(adapter_request)
        self._record_usage(request.tier, response.model, response.usage, request.caller)
        return response

    async def generate_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        adapter = self._adapters[request.tier]
        resolved_model = request.model or self._models.get(
            request.tier, adapter.default_model
        )
        resolved_effort = request.reasoning_effort or self._reasoning_efforts.get(
            request.tier
        )
        adapter_request = replace(
            request,
            model=resolved_model,
            reasoning_effort=resolved_effort,
        )
        logger.debug(
            "llm_route_tools",
            tier=request.tier,
            adapter=adapter.name,
            model=resolved_model,
            reasoning_effort=resolved_effort,
            caller=request.caller,
        )
        response = await adapter.generate_with_tools(adapter_request)
        self._record_usage(request.tier, response.model, response.usage, request.caller)
        return response

    async def describe_images(self, request: LLMVisionRequest) -> LLMResponse:
        if self._vision is None:
            return LLMResponse(
                text="Не могу видеть фото, Gemini API не настроен",
                model="",
                usage=LLMUsage(),
            )
        effective = (
            request
            if request.prompt
            else replace(request, prompt=DEFAULT_VISION_PROMPT)
        )
        response = await self._vision.describe_images(effective)
        if self._usage is not None:
            self._usage.record_gemini_call(
                provider=self._vision_provider,
                model=response.model,
                caller=effective.caller,
            )
        return response

    async def generate_image(self, request: LLMImageRequest) -> LLMImageResponse:
        if self._image_generator is None:
            raise ProviderUnavailableError("Image generation is disabled")
        generator = self._image_generator
        if callable(generator) and not hasattr(type(generator), "generate_image"):
            response = cast(
                "LLMImageResponse",
                await cast("Any", generator)(request),
            )
        else:
            response = await cast("Any", generator).generate_image(request)
        if self._usage is not None:
            self._usage.record_api_call(
                provider=self._image_provider or "image_generation",
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                cache_write_tokens=response.usage.cache_write_tokens,
                caller=request.caller,
            )
        return response

    def set_usage_tracker(self, tracker: UsageTracker) -> None:
        """Attach usage tracker after construction (bot bootstrap path)."""
        self._usage = tracker

    def get_adapter(self, tier: Tier) -> BaseLLMAdapter:
        return self._adapters[tier]

    async def generate_oneoff(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        system: str = "",
        caller: str = "",
    ) -> LLMResponse:
        """Dispatch a single request to an explicit ``(provider, model)`` pair.

        Bypasses tier mapping. Built for ad-hoc destinations like the
        /compare command, where the caller picks both ends of the A/B
        and the existing tier wiring is not the right axis.

        Adapter instances are cached per-router so repeat calls don't
        spin up a new HTTP client every time.

        Raises:
            LLMError: provider is not in PROVIDERS, or its required setting
                is missing, or the underlying adapter call fails.
        """
        spec = PROVIDERS.get(provider)
        if spec is None:
            raise LLMError(
                f"Unknown LLM provider {provider!r}. Allowed: {sorted(PROVIDERS)}"
            )
        if spec.requires_setting:
            from src.core.config import get_settings

            if not getattr(get_settings(), spec.requires_setting):
                raise LLMError(
                    f"{provider} requires {spec.requires_setting.upper()} in .env"
                )
        if provider not in self._oneoff_adapters:
            self._oneoff_adapters[provider] = spec.adapter_factory()
        adapter = self._oneoff_adapters[provider]

        resolved = resolve_model(provider, model)
        api_id = resolved.api_id if resolved is not None else model

        request = LLMRequest(
            prompt=prompt,
            system=system,
            model=api_id,
            caller=caller,
        )
        logger.info(
            "llm_oneoff",
            provider=provider,
            model=api_id,
            caller=caller,
        )
        response = await adapter.generate(request)
        self._record_provider_usage(provider, response.model, response.usage, caller)
        return response

    def _record_usage(
        self,
        tier: Tier,
        model: str,
        usage: LLMUsage,
        caller: str,
    ) -> None:
        """Dispatch by provider's ``billing_mode`` for tier-routed calls."""
        provider = self._providers_by_tier.get(tier, "")
        self._record_provider_usage(provider, model, usage, caller)

    def _record_provider_usage(
        self,
        provider: str,
        model: str,
        usage: LLMUsage,
        caller: str,
    ) -> None:
        """Provider-keyed usage recording, shared by tier routing and oneoff.

        Per-token (Anthropic API, OpenRouter): record full token counts and costs.
        Subscription CLI: record one call, no money.
        Free tier (Gemini): record one call, no money.
        Unknown providers route to record_api_call as a best-effort default.
        """
        if self._usage is None:
            return
        spec = PROVIDERS.get(provider)
        if spec is None or spec.billing_mode == "per_token":
            self._usage.record_api_call(
                provider=provider or "anthropic_api",
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                caller=caller,
            )
        elif spec.billing_mode == "subscription":
            self._usage.record_cli_call(provider=provider, model=model, caller=caller)
        elif spec.billing_mode == "free_tier":
            self._usage.record_gemini_call(
                provider=provider, model=model, caller=caller
            )


def _validate_tier_config(
    tier: Tier,
    provider_name: str,
    model_name: str,
    claude_cli_path: str,
    codex_cli_path: str,
) -> None:
    """Fail-fast validation for one tier's (provider, model) pair."""
    spec = PROVIDERS.get(provider_name)
    if spec is None:
        raise RuntimeError(
            f"Unknown LLM provider for {tier}: {provider_name!r}. "
            f"Allowed: {sorted(PROVIDERS)}"
        )
    if provider_name == "claude_cli" and not claude_cli_path:
        raise RuntimeError("CLAUDE_CLI_PATH is required when any tier uses claude_cli")
    if provider_name == "codex_cli" and not codex_cli_path:
        raise RuntimeError("CODEX_CLI_PATH is required when any tier uses codex_cli")
    if (
        spec.billing_mode == "per_token"
        and resolve_model(provider_name, model_name) is None
    ):
        aliases = sorted({a for m in spec.models.values() for a in m.aliases})
        raise RuntimeError(
            f"{tier} → provider={provider_name!r} unknown model "
            f"{model_name!r}; known: {sorted(spec.models)} + aliases {aliases}"
        )


def create_router() -> LLMRouter:
    """Build a router from current settings.

    Each tier is configured by a pair (``WORKER_PROVIDER``/``WORKER_MODEL``,
    etc.) read from ``Settings``. ``PROVIDERS`` (in ``src.llm.providers``)
    is the source of truth for which models exist under each provider and
    how they bill. Validation is fail-fast: unknown provider, unknown model
    (for per_token providers), or missing API key all raise ``RuntimeError``
    at startup rather than producing a silent fallback.
    """
    settings = get_settings()

    adapter_cache: dict[str, BaseLLMAdapter] = {}
    adapters: dict[Tier, BaseLLMAdapter] = {}
    models: dict[Tier, str] = {}
    reasoning_efforts: dict[Tier, ReasoningEffort] = {}
    providers_by_tier: dict[Tier, str] = {}

    for tier, (prov_attr, model_attr) in TIER_KEYS.items():
        provider_name = getattr(settings, prov_attr)
        model_name = getattr(settings, model_attr)
        _validate_tier_config(
            tier,
            provider_name,
            model_name,
            settings.claude_cli_path,
            settings.codex_cli_path,
        )
        spec = PROVIDERS[provider_name]
        if spec.requires_setting and not getattr(settings, spec.requires_setting):
            raise RuntimeError(
                f"{tier} → provider={provider_name!r} requires "
                f"{spec.requires_setting.upper()} in .env"
            )
        if provider_name not in adapter_cache:
            adapter_cache[provider_name] = spec.adapter_factory()
        adapters[tier] = adapter_cache[provider_name]
        resolved = resolve_model(provider_name, model_name)
        models[tier] = resolved.api_id if resolved is not None else model_name
        reasoning_efforts[tier] = getattr(settings, TIER_REASONING_KEYS[tier])
        providers_by_tier[tier] = provider_name

    vision_provider_name = settings.vision_provider
    vision_adapter = _build_vision_adapter(
        settings, vision_provider_name, adapter_cache
    )
    image_generator = _build_image_generator(settings)

    logger.info(
        "llm_router_created",
        worker=f"{providers_by_tier['worker']}/{models['worker']}",
        analyst=f"{providers_by_tier['analyst']}/{models['analyst']}",
        strategist=f"{providers_by_tier['strategist']}/{models['strategist']}",
        strategist_reasoning_effort=reasoning_efforts["strategist"],
        vision=f"{vision_provider_name}/{settings.vision_model}"
        if vision_adapter
        else "disabled",
        image_generation=(
            f"{settings.image_generation_provider}/{settings.image_generation_model}"
            if image_generator is not None
            else "disabled"
        ),
    )

    return LLMRouter(
        adapters,
        models,
        reasoning_efforts=reasoning_efforts,
        providers_by_tier=providers_by_tier,
        vision_adapter=vision_adapter,
        vision_provider=vision_provider_name,
        image_generator=image_generator,
        image_provider=settings.image_generation_provider,
    )


def _build_vision_adapter(
    settings: object,
    vision_provider_name: str,
    adapter_cache: dict[str, BaseLLMAdapter],
) -> GeminiAdapter | None:
    """Wire the optional vision adapter; returns None when its provider's
    required setting is missing (e.g. no GOOGLE_API_KEY)."""
    if not vision_provider_name:
        return None
    vspec = PROVIDERS.get(vision_provider_name)
    if vspec is None:
        raise RuntimeError(
            f"Unknown vision provider: {vision_provider_name!r}. "
            f"Allowed: {sorted(PROVIDERS)}"
        )
    if vspec.requires_setting and not getattr(settings, vspec.requires_setting):
        return None
    if vision_provider_name not in adapter_cache:
        adapter_cache[vision_provider_name] = vspec.adapter_factory()
    from src.llm.gemini import GeminiAdapter as _GeminiAdapter

    built = adapter_cache[vision_provider_name]
    return built if isinstance(built, _GeminiAdapter) else None


def _build_image_generator(settings: Settings) -> ImageGeneratorProtocol | None:
    """Wire optional image generation; safe/off by default."""
    if not settings.image_generation_enabled:
        return None
    provider = settings.image_generation_provider.strip()
    if provider == "cli":
        command = settings.image_generation_cli_command.strip()
        if not command:
            raise RuntimeError(
                "IMAGE_GENERATION_CLI_COMMAND is required when CLI image "
                "generation is enabled"
            )
        from src.llm.cli_images import CLIImageGenerator

        return CLIImageGenerator(
            command=command,
            model=settings.image_generation_model.strip(),
            size=settings.image_generation_size,
            timeout_seconds=settings.image_generation_cli_timeout_seconds,
        )
    if provider == "openai":
        model = settings.image_generation_model.strip()
        api_key = settings.openai_api_key.strip()
        if not model:
            raise RuntimeError("IMAGE_GENERATION_MODEL is required when enabled")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when image generation is enabled"
            )
        from src.llm.openai_images import OpenAIImageGenerator

        return OpenAIImageGenerator(
            api_key=api_key,
            model=model,
            size=settings.image_generation_size,
        )
    raise RuntimeError(f"Unknown image generation provider: {provider!r}")


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Get or create the singleton LLM router."""
    global _router
    if _router is None:
        _router = create_router()
    return _router
