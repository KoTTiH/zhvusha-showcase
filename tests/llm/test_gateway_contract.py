"""Contract tests for the LLM Gateway (phase 2).

Verifies that ``LLMRouter`` conforms to ``LLMGatewayProtocol`` and that all
public types in ``src.llm.protocols`` have the expected shape. Uses
``AsyncMock`` for adapters — **no** real LLM calls happen here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.llm.protocols import (
    DEFAULT_VISION_PROMPT,
    AuthenticationError,
    BudgetExceededError,
    LLMError,
    LLMGatewayProtocol,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
    LLMVisionRequest,
    ProviderUnavailableError,
    RateLimitError,
    Tier,
    ToolDefinition,
)
from src.llm.router import LLMRouter

pytestmark = pytest.mark.contract


def _mock_adapter(name: str = "mock", default_model: str = "mock-model") -> MagicMock:
    adapter = MagicMock()
    adapter.name = name
    adapter.default_model = default_model
    adapter.generate = AsyncMock()
    adapter.generate_with_tools = AsyncMock()
    return adapter


def _build_router(
    adapter: MagicMock | None = None,
    *,
    vision: MagicMock | None = None,
    tracker: MagicMock | None = None,
    providers_by_tier: dict[str, str] | None = None,
) -> LLMRouter:
    adapter = adapter or _mock_adapter()
    return LLMRouter(
        adapters={
            "worker": adapter,
            "analyst": adapter,
            "strategist": adapter,
        },
        models={
            "worker": "mock-worker",
            "analyst": "mock-analyst",
            "strategist": "mock-strategist",
        },
        providers_by_tier=providers_by_tier,  # type: ignore[arg-type]
        vision_adapter=vision,
        usage_tracker=tracker,
    )


# === Protocol compliance ===


class TestProtocolCompliance:
    def test_llm_router_is_instance_of_protocol(self) -> None:
        """``LLMRouter`` is a runtime-checkable ``LLMGatewayProtocol`` instance."""
        router = _build_router()
        assert isinstance(router, LLMGatewayProtocol)

    def test_tier_exported_from_protocols(self) -> None:
        """``Tier`` is re-exported from ``src.llm.protocols`` for convenience."""
        # Tier is a Literal type — we check its constituent values roundtrip.
        assert Tier.__args__ == ("worker", "analyst", "strategist")  # type: ignore[attr-defined]


# === Data class shapes ===


class TestLLMRequest:
    def test_is_frozen(self) -> None:
        req = LLMRequest(prompt="test")
        with pytest.raises((AttributeError, TypeError)):
            req.prompt = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        req = LLMRequest(prompt="x")
        assert req.system == ""
        assert req.tier == "worker"
        assert req.model is None
        assert req.temperature is None
        assert req.caller == ""

    def test_explicit_fields(self) -> None:
        req = LLMRequest(
            prompt="hi",
            system="sys",
            tier="analyst",
            model="sonnet",
            temperature=0.7,
            caller="test",
        )
        assert req.tier == "analyst"
        assert req.temperature == 0.7
        assert req.caller == "test"


class TestLLMResponse:
    def test_shape(self) -> None:
        resp = LLMResponse(text="hello", model="haiku", usage=LLMUsage(input_tokens=10))
        assert resp.text == "hello"
        assert resp.model == "haiku"
        assert resp.usage.input_tokens == 10

    def test_usage_default(self) -> None:
        resp = LLMResponse(text="x", model="y")
        assert resp.usage == LLMUsage()

    def test_is_frozen(self) -> None:
        resp = LLMResponse(text="x", model="y")
        with pytest.raises((AttributeError, TypeError)):
            resp.text = "mutated"  # type: ignore[misc]


class TestLLMUsage:
    def test_all_fields_default_zero(self) -> None:
        u = LLMUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0


class TestLLMMessage:
    def test_shape(self) -> None:
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestToolDefinition:
    def test_required_fields(self) -> None:
        td = ToolDefinition(
            name="search", description="Search", input_schema={"type": "object"}
        )
        assert td.name == "search"
        assert td.description == "Search"
        assert td.input_schema == {"type": "object"}


class TestLLMToolRequestResponse:
    def test_tool_request_shape(self) -> None:
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
        )
        assert req.tier == "analyst"
        assert len(req.tools) == 1

    def test_tool_response_shape(self) -> None:
        resp = LLMToolResponse(
            content_blocks=[{"type": "text"}],
            stop_reason="end_turn",
            model="sonnet",
            usage=LLMUsage(input_tokens=5),
        )
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 5


class TestLLMVisionRequest:
    def test_shape(self) -> None:
        req = LLMVisionRequest(images=[b"\xff\xd8"], prompt="describe")
        assert req.images == [b"\xff\xd8"]
        assert req.prompt == "describe"

    def test_default_prompt(self) -> None:
        req = LLMVisionRequest(images=[b"x"])
        assert req.prompt == DEFAULT_VISION_PROMPT


# === Errors ===


class TestErrorHierarchy:
    def test_llm_error_is_exception(self) -> None:
        assert issubclass(LLMError, Exception)

    def test_rate_limit_is_llm_error(self) -> None:
        assert issubclass(RateLimitError, LLMError)

    def test_budget_exceeded_is_llm_error(self) -> None:
        assert issubclass(BudgetExceededError, LLMError)

    def test_authentication_is_llm_error(self) -> None:
        assert issubclass(AuthenticationError, LLMError)

    def test_provider_unavailable_is_llm_error(self) -> None:
        assert issubclass(ProviderUnavailableError, LLMError)


# === Router behavior against the protocol ===


class TestGenerateHappyPath:
    async def test_returns_llm_response(self) -> None:
        adapter = _mock_adapter()
        adapter.generate.return_value = LLMResponse(
            text="hello",
            model="mock-analyst",
            usage=LLMUsage(input_tokens=5, output_tokens=10),
        )
        router = _build_router(adapter)

        resp = await router.generate(
            LLMRequest(prompt="hi", tier="analyst", caller="test")
        )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello"
        assert resp.usage.input_tokens == 5

    async def test_resolves_tier_to_model(self) -> None:
        adapter = _mock_adapter()
        adapter.generate.return_value = LLMResponse(
            text="", model="mock-analyst", usage=LLMUsage()
        )
        router = _build_router(adapter)

        await router.generate(LLMRequest(prompt="x", tier="analyst"))

        call_arg = adapter.generate.call_args.args[0]
        assert isinstance(call_arg, LLMRequest)
        assert call_arg.model == "mock-analyst"
        assert call_arg.tier == "analyst"

    async def test_caller_override_model(self) -> None:
        adapter = _mock_adapter()
        adapter.generate.return_value = LLMResponse(
            text="", model="custom", usage=LLMUsage()
        )
        router = _build_router(adapter)

        await router.generate(LLMRequest(prompt="x", tier="analyst", model="custom"))

        call_arg = adapter.generate.call_args.args[0]
        assert call_arg.model == "custom"


class TestGenerateErrors:
    async def test_adapter_error_propagates(self) -> None:
        adapter = _mock_adapter()
        adapter.generate.side_effect = LLMError("provider down")
        router = _build_router(adapter)

        with pytest.raises(LLMError, match="provider down"):
            await router.generate(LLMRequest(prompt="x"))


class TestGenerateWithTools:
    async def test_returns_tool_response(self) -> None:
        adapter = _mock_adapter()
        adapter.generate_with_tools.return_value = LLMToolResponse(
            content_blocks=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
            model="mock-analyst",
            usage=LLMUsage(),
        )
        router = _build_router(adapter)

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
        )
        resp = await router.generate_with_tools(req)

        assert isinstance(resp, LLMToolResponse)
        assert resp.stop_reason == "end_turn"
        # Adapter received a request with resolved model
        call_arg = adapter.generate_with_tools.call_args.args[0]
        assert isinstance(call_arg, LLMToolRequest)
        assert call_arg.model == "mock-analyst"


class TestDescribeImages:
    async def test_no_vision_adapter_returns_fallback(self) -> None:
        router = _build_router(vision=None)
        resp = await router.describe_images(LLMVisionRequest(images=[b"\xff\xd8"]))
        assert isinstance(resp, LLMResponse)
        assert "не настроен" in resp.text.lower() or "gemini" in resp.text.lower()

    async def test_delegates_to_vision_adapter(self) -> None:
        vision = AsyncMock()
        vision.describe_images = AsyncMock(
            return_value=LLMResponse(
                text="A cat on a couch", model="gemini-vision", usage=LLMUsage()
            )
        )
        router = _build_router(vision=vision)

        resp = await router.describe_images(LLMVisionRequest(images=[b"img"]))
        assert resp.text == "A cat on a couch"
        vision.describe_images.assert_awaited_once()


class TestUsageTrackerIntegration:
    async def test_anthropic_usage_forwarded_to_tracker(self) -> None:
        adapter = _mock_adapter(name="anthropic_api")
        adapter.generate.return_value = LLMResponse(
            text="x",
            model="claude-sonnet-4-6",
            usage=LLMUsage(input_tokens=100, output_tokens=50, cache_read_tokens=1000),
        )
        tracker = MagicMock()
        router = _build_router(
            adapter,
            tracker=tracker,
            providers_by_tier={
                "worker": "anthropic_api",
                "analyst": "anthropic_api",
                "strategist": "claude_cli",
            },
        )

        await router.generate(
            LLMRequest(prompt="x", tier="analyst", caller="test_caller")
        )

        tracker.record_api_call.assert_called_once_with(
            provider="anthropic_api",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=1000,
            cache_write_tokens=0,
            caller="test_caller",
        )

    async def test_claude_cli_usage_recorded_coarsely(self) -> None:
        adapter = _mock_adapter(name="claude_cli")
        adapter.generate.return_value = LLMResponse(
            text="x", model="sonnet", usage=LLMUsage()
        )
        tracker = MagicMock()
        router = _build_router(
            adapter,
            tracker=tracker,
            providers_by_tier={
                "worker": "claude_cli",
                "analyst": "claude_cli",
                "strategist": "claude_cli",
            },
        )

        await router.generate(LLMRequest(prompt="x", tier="strategist"))

        tracker.record_cli_call.assert_called_once()
