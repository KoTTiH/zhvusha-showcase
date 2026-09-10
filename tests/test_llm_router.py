import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.gemini import GeminiAdapter
from src.llm.protocols import (
    LLMGatewayProtocol,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
    LLMVisionRequest,
    ToolDefinition,
)
from src.llm.router import LLMRouter, create_router


class FakeAdapter(BaseLLMAdapter):
    name: str = "fake"
    default_model: str = "fake-v1"

    def __init__(self) -> None:
        self.last_call: dict = {}
        self.usage = LLMUsage()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_call = {
            "prompt": request.prompt,
            "system": request.system,
            "model": request.model,
            "reasoning_effort": request.reasoning_effort,
            "temperature": request.temperature,
            "tier": request.tier,
            "caller": request.caller,
        }
        return LLMResponse(
            text=f"fake:{request.prompt}",
            model=request.model or self.default_model,
            usage=self.usage,
        )


@pytest.fixture
def fake_adapters():
    worker = FakeAdapter()
    analyst = FakeAdapter()
    strategist = FakeAdapter()
    return {"worker": worker, "analyst": analyst, "strategist": strategist}


@pytest.fixture
def fake_models():
    return {"worker": "haiku", "analyst": "sonnet", "strategist": "opus"}


def _req(
    prompt: str = "test",
    *,
    system: str = "",
    tier: str = "worker",
    model: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    caller: str = "",
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system=system,
        tier=tier,  # type: ignore[arg-type]
        model=model,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        temperature=temperature,
        caller=caller,
    )


async def test_router_implements_protocol(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)
    assert isinstance(router, LLMGatewayProtocol)


async def test_router_dispatches_by_tier(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)

    result = await router.generate(_req("hello", tier="analyst"))

    assert isinstance(result, LLMResponse)
    assert result.text == "fake:hello"
    assert fake_adapters["analyst"].last_call["prompt"] == "hello"
    assert fake_adapters["worker"].last_call == {}


async def test_router_passes_system_and_model(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(
        _req("test", system="Be brief", tier="worker", model="custom")
    )

    call = fake_adapters["worker"].last_call
    assert call["system"] == "Be brief"
    assert call["model"] == "custom"


async def test_router_defaults_to_worker(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(_req("test"))

    assert fake_adapters["worker"].last_call["prompt"] == "test"


async def test_router_injects_tier_model(fake_adapters, fake_models):
    """When model=None, router injects the tier's configured model."""
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(_req("test", tier="strategist"))

    assert fake_adapters["strategist"].last_call["model"] == "opus"


async def test_router_injects_tier_reasoning_effort(fake_adapters, fake_models):
    router = LLMRouter(
        fake_adapters,
        fake_models,
        reasoning_efforts={
            "worker": "medium",
            "analyst": "high",
            "strategist": "xhigh",
        },
    )

    await router.generate(_req("test", tier="strategist"))

    assert fake_adapters["strategist"].last_call["reasoning_effort"] == "xhigh"


async def test_router_caller_model_overrides_tier(fake_adapters, fake_models):
    """Caller-supplied model takes precedence over tier default."""
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(_req("test", tier="worker", model="custom-model"))

    assert fake_adapters["worker"].last_call["model"] == "custom-model"


async def test_router_caller_reasoning_effort_overrides_tier(
    fake_adapters, fake_models
):
    router = LLMRouter(
        fake_adapters,
        fake_models,
        reasoning_efforts={"worker": "medium"},
    )

    await router.generate(_req("test", tier="worker", reasoning_effort="xhigh"))

    assert fake_adapters["worker"].last_call["reasoning_effort"] == "xhigh"


async def test_router_passes_token_usage_to_tracker(fake_models):
    """Router reads LLMResponse.usage and passes it to tracker with the
    provider taken from providers_by_tier (per_token billing dispatch)."""
    adapter = FakeAdapter()
    adapter.name = "anthropic_api"  # type: ignore[assignment]
    adapter.usage = LLMUsage(
        input_tokens=500,
        output_tokens=200,
        cache_read_tokens=1000,
        cache_write_tokens=0,
    )
    adapters = {"worker": adapter, "analyst": adapter, "strategist": adapter}
    providers = {
        "worker": "anthropic_api",
        "analyst": "anthropic_api",
        "strategist": "claude_cli",
    }

    tracker = MagicMock()
    router = LLMRouter(
        adapters, fake_models, providers_by_tier=providers, usage_tracker=tracker
    )

    await router.generate(_req("test", tier="analyst"))

    tracker.record_api_call.assert_called_once_with(
        provider="anthropic_api",
        model="sonnet",
        input_tokens=500,
        output_tokens=200,
        cache_read_tokens=1000,
        cache_write_tokens=0,
        caller="",
    )


def test_get_adapter(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)

    assert router.get_adapter("strategist") is fake_adapters["strategist"]


async def test_describe_images_delegates_to_vision(fake_adapters, fake_models):
    """describe_images routes to the vision adapter with an LLMVisionRequest."""
    vision = AsyncMock(spec=GeminiAdapter)
    vision.describe_images = AsyncMock(
        return_value=LLMResponse(
            text="A cat sitting on a table",
            model="gemini-2.5-flash-lite",
            usage=LLMUsage(),
        )
    )
    router = LLMRouter(fake_adapters, fake_models, vision_adapter=vision)

    result = await router.describe_images(LLMVisionRequest(images=[b"img-data"]))

    assert isinstance(result, LLMResponse)
    assert result.text == "A cat sitting on a table"
    vision.describe_images.assert_awaited_once()
    passed = vision.describe_images.call_args[0][0]
    assert isinstance(passed, LLMVisionRequest)
    assert passed.images == [b"img-data"]


async def test_describe_images_no_vision_returns_message(fake_adapters, fake_models):
    """Without vision adapter, describe_images returns a helpful LLMResponse."""
    router = LLMRouter(fake_adapters, fake_models, vision_adapter=None)

    result = await router.describe_images(LLMVisionRequest(images=[b"img-data"]))

    assert isinstance(result, LLMResponse)
    assert "Gemini API не настроен" in result.text


_BASE_ENV = {
    "BOT_TOKEN": "fake",
    "CHANNEL_ID": "@test",
    "ADMIN_USER_ID": "1",
    "CLAUDE_CLI_PATH": "claude",
    "CODEX_CLI_PATH": "codex",
}


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "claude_cli",
        "ANALYST_PROVIDER": "claude_cli",
        "STRATEGIST_PROVIDER": "claude_cli",
        "WORKER_MODEL": "haiku",
        "ANALYST_MODEL": "sonnet",
        "STRATEGIST_MODEL": "opus",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
def test_create_router_all_tiers_via_claude_cli():
    """Provider routing: every tier explicitly bound to claude_cli."""
    get_settings.cache_clear()
    router = create_router()

    assert router.get_adapter("worker").name == "claude_cli"
    assert router.get_adapter("analyst").name == "claude_cli"
    assert router.get_adapter("strategist").name == "claude_cli"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "WORKER_PROVIDER": "codex_cli",
        "ANALYST_PROVIDER": "codex_cli",
        "STRATEGIST_PROVIDER": "codex_cli",
        "WORKER_MODEL": "default",
        "ANALYST_MODEL": "default",
        "STRATEGIST_MODEL": "default",
        "GOOGLE_API_KEY": "",
    },
    clear=True,
)
def test_create_router_defaults_to_codex_cli_subscription():
    """Default text tiers use Codex CLI subscription and need no API key."""
    get_settings.cache_clear()
    router = create_router()

    assert router.get_adapter("worker").name == "codex_cli"
    assert router.get_adapter("analyst").name == "codex_cli"
    assert router.get_adapter("strategist").name == "codex_cli"
    assert router._models == {
        "worker": "default",
        "analyst": "default",
        "strategist": "default",
    }
    assert router._reasoning_efforts["strategist"] == "xhigh"


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "CODEX_CLI_PATH": "",
        "WORKER_PROVIDER": "codex_cli",
        "ANALYST_PROVIDER": "codex_cli",
        "STRATEGIST_PROVIDER": "codex_cli",
    },
    clear=True,
)
def test_create_router_codex_provider_without_codex_path_raises():
    """CODEX_CLI_PATH empty + any tier on codex_cli → fail-fast."""
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="CODEX_CLI_PATH is required"):
        create_router()


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "anthropic_api",
        "WORKER_MODEL": "haiku",
        "ANALYST_PROVIDER": "anthropic_api",
        "ANALYST_MODEL": "sonnet",
        "STRATEGIST_PROVIDER": "claude_cli",
        "STRATEGIST_MODEL": "opus",
        "ANTHROPIC_API_KEY": "fake_anthropic_test_key",
        "GOOGLE_API_KEY": "test-key",
    },
    clear=True,
)
def test_create_router_provider_per_tier_split():
    """Default-style mix: API for worker/analyst, CLI for strategist."""
    get_settings.cache_clear()
    router = create_router()

    assert router.get_adapter("worker").name == "anthropic_api"
    assert router.get_adapter("analyst").name == "anthropic_api"
    assert router.get_adapter("strategist").name == "claude_cli"


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "anthropic_api",
        "WORKER_MODEL": "haiku",
        "ANALYST_PROVIDER": "anthropic_api",
        "ANALYST_MODEL": "sonnet",
        "STRATEGIST_PROVIDER": "anthropic_api",
        "STRATEGIST_MODEL": "opus",
        "ANTHROPIC_API_KEY": "fake_anthropic_test_key",
    },
    clear=True,
)
def test_create_router_strategist_via_anthropic_api():
    """Used to be impossible (strategist hard-coded to CLI). Now controlled
    by STRATEGIST_PROVIDER alone — that's the principle №1 smoke."""
    get_settings.cache_clear()
    router = create_router()

    assert router.get_adapter("strategist").name == "anthropic_api"


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "claude_cli",
        "ANALYST_PROVIDER": "claude_cli",
        "STRATEGIST_PROVIDER": "claude_cli",
        "GOOGLE_API_KEY": "",
    },
    clear=True,
)
def test_create_router_without_gemini_vision_disabled():
    """No GOOGLE_API_KEY → vision adapter omitted, text tiers still work."""
    get_settings.cache_clear()
    router = create_router()

    assert router.get_adapter("worker").name == "claude_cli"
    assert router._vision is None


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "CLAUDE_CLI_PATH": "",
        "WORKER_PROVIDER": "claude_cli",
        "ANALYST_PROVIDER": "claude_cli",
        "STRATEGIST_PROVIDER": "claude_cli",
    },
    clear=True,
)
def test_create_router_cli_provider_without_claude_raises():
    """CLAUDE_CLI_PATH empty + any tier on claude_cli → fail-fast."""
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="CLAUDE_CLI_PATH is required"):
        create_router()


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "openai",
    },
    clear=True,
)
def test_create_router_unknown_provider_raises():
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        create_router()


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "anthropic_api",
        "WORKER_MODEL": "missing_model_for_tests",
        "ANTHROPIC_API_KEY": "fake_anthropic_test_key",
    },
    clear=True,
)
def test_create_router_unknown_model_for_per_token_raises():
    """For per_token providers, model must exist in the registry —
    otherwise tokens go to the API and the call fails late."""
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="unknown model"):
        create_router()


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "anthropic_api",
        "WORKER_MODEL": "haiku",
        "ANTHROPIC_API_KEY": "",
    },
    clear=True,
)
def test_create_router_missing_required_setting_raises():
    """ANTHROPIC_API_KEY required if any tier uses anthropic_api."""
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        create_router()


@patch.dict(
    os.environ,
    {
        **_BASE_ENV,
        "WORKER_PROVIDER": "anthropic_api",
        "WORKER_MODEL": "haiku",
        "ANALYST_PROVIDER": "anthropic_api",
        "ANALYST_MODEL": "sonnet",
        "STRATEGIST_PROVIDER": "claude_cli",
        "STRATEGIST_MODEL": "opus",
        "ANTHROPIC_API_KEY": "fake_anthropic_test_key",
    },
    clear=True,
)
def test_create_router_resolves_alias_to_api_id():
    """Router stores the resolved api_id (not the alias) so adapters
    receive a provider-native id."""
    get_settings.cache_clear()
    router = create_router()

    assert router._models["worker"] == "claude-haiku-4-5-20251001"


async def test_router_cli_usage_tracking(fake_models):
    """Subscription billing → record_cli_call (regardless of adapter.name)."""
    adapter = FakeAdapter()
    adapter.name = "claude_cli"  # type: ignore[assignment]
    adapters = {"worker": adapter, "analyst": adapter, "strategist": adapter}
    providers = {
        "worker": "claude_cli",
        "analyst": "claude_cli",
        "strategist": "claude_cli",
    }

    tracker = MagicMock()
    router = LLMRouter(
        adapters, fake_models, providers_by_tier=providers, usage_tracker=tracker
    )

    await router.generate(_req("test", tier="worker"))

    tracker.record_cli_call.assert_called_once()


async def test_generate_with_tools_returns_tool_response(fake_models):
    """generate_with_tools routes to the correct adapter and returns a typed response."""
    adapter = AsyncMock()
    adapter.name = "anthropic_api"
    adapter.default_model = "claude-sonnet-4-6"
    adapter.generate_with_tools = AsyncMock(
        return_value=LLMToolResponse(
            content_blocks=[{"type": "text"}],
            stop_reason="end_turn",
            model="sonnet",
            usage=LLMUsage(input_tokens=100, output_tokens=50),
        )
    )
    adapters = {"worker": adapter, "analyst": adapter, "strategist": adapter}
    providers = {
        "worker": "anthropic_api",
        "analyst": "anthropic_api",
        "strategist": "claude_cli",
    }

    tracker = MagicMock()
    router = LLMRouter(
        adapters, fake_models, providers_by_tier=providers, usage_tracker=tracker
    )

    req = LLMToolRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            ToolDefinition(
                name="t",
                description="d",
                input_schema={"type": "object"},
            )
        ],
        tier="analyst",
        caller="test",
    )
    response = await router.generate_with_tools(req)

    assert isinstance(response, LLMToolResponse)
    assert response.stop_reason == "end_turn"
    adapter.generate_with_tools.assert_awaited_once()
    tracker.record_api_call.assert_called_once()


async def test_describe_images_tracks_usage(fake_adapters, fake_models):
    """Gemini usage is tracked when vision adapter is present."""
    vision = AsyncMock(spec=GeminiAdapter)
    vision.describe_images = AsyncMock(
        return_value=LLMResponse(text="A cat", model="gem", usage=LLMUsage())
    )
    tracker = MagicMock()
    router = LLMRouter(
        fake_adapters, fake_models, vision_adapter=vision, usage_tracker=tracker
    )

    await router.describe_images(LLMVisionRequest(images=[b"img"]))

    tracker.record_gemini_call.assert_called_once()


def test_set_usage_tracker(fake_adapters, fake_models):
    router = LLMRouter(fake_adapters, fake_models)
    assert router._usage is None
    tracker = MagicMock()
    router.set_usage_tracker(tracker)
    assert router._usage is tracker


async def test_router_passes_temperature_to_adapter(fake_adapters, fake_models):
    """Router passes temperature through to adapter."""
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(_req("test", tier="analyst", temperature=0.0))

    call = fake_adapters["analyst"].last_call
    assert call["temperature"] == 0.0


async def test_router_temperature_none_by_default(fake_adapters, fake_models):
    """Temperature is None by default."""
    router = LLMRouter(fake_adapters, fake_models)

    await router.generate(_req("test", tier="analyst"))

    call = fake_adapters["analyst"].last_call
    assert call["temperature"] is None


async def test_router_caller_propagates(fake_adapters, fake_models):
    """Caller is propagated from request to adapter (via logs / usage tracker)."""
    adapter = FakeAdapter()
    adapter.name = "anthropic_api"  # type: ignore[assignment]
    adapters = {"worker": adapter, "analyst": adapter, "strategist": adapter}
    providers = {
        "worker": "anthropic_api",
        "analyst": "anthropic_api",
        "strategist": "claude_cli",
    }

    tracker = MagicMock()
    router = LLMRouter(
        adapters, fake_models, providers_by_tier=providers, usage_tracker=tracker
    )

    await router.generate(_req("test", tier="analyst", caller="decision_plan"))

    tracker.record_api_call.assert_called_once()
    assert tracker.record_api_call.call_args.kwargs["caller"] == "decision_plan"
