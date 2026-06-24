"""Tests for AnthropicAPIAdapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.llm.anthropic_api import AnthropicAPIAdapter
from src.llm.protocols import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    ToolDefinition,
)


@pytest.fixture
def adapter() -> AnthropicAPIAdapter:
    return AnthropicAPIAdapter()


def _req(
    prompt: str = "hello",
    *,
    system: str = "",
    model: str | None = "claude-sonnet-4-6",
    temperature: float | None = None,
) -> LLMRequest:
    """Default model = sonnet api_id because the router resolves alias→api_id
    before dispatch. Adapter tests treat ``model`` as authoritative."""
    return LLMRequest(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        tier="analyst",
        caller="test",
    )


def _tool_req(
    messages: list[dict],
    tools: list[ToolDefinition],
    *,
    system: str = "",
    temperature: float | None = None,
    model: str | None = "claude-sonnet-4-6",
) -> LLMToolRequest:
    return LLMToolRequest(
        messages=messages,
        tools=tools,
        system=system,
        temperature=temperature,
        tier="analyst",
        caller="test",
        model=model,
    )


def _mock_usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    text_block = SimpleNamespace(text="Generated response")
    response = SimpleNamespace(content=[text_block], usage=_mock_usage())
    client.messages.create = AsyncMock(return_value=response)
    return client


def test_adapter_rejects_empty_model() -> None:
    """The router resolves api_id before dispatch; the adapter has no
    fallback table, so calling it without a model is a programming bug."""
    a = AnthropicAPIAdapter()
    a._client = AsyncMock()
    import asyncio

    with pytest.raises(LLMError, match="empty model"):
        asyncio.run(a.generate(_req("hi", model=None)))


async def test_generate_returns_response(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    adapter._client = mock_client

    result = await adapter.generate(_req("hello", model="claude-sonnet-4-6"))

    assert isinstance(result, LLMResponse)
    assert result.text == "Generated response"
    assert result.model == "claude-sonnet-4-6"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    mock_client.messages.create.assert_awaited_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert "system" not in call_kwargs  # empty system not passed


async def test_generate_with_system_sends_cache_control(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    adapter._client = mock_client

    await adapter.generate(_req("hello", system="Be brief"))

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert isinstance(system_blocks, list)
    assert len(system_blocks) == 1
    assert system_blocks[0]["type"] == "text"
    assert system_blocks[0]["text"] == "Be brief"
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


async def test_generate_with_model_override(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """Adapter passes the api_id straight through — alias resolution lives
    in the router, not here."""
    adapter._client = mock_client

    await adapter.generate(_req("hello", model="claude-haiku-4-5-20251001"))

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


async def test_api_error_raises_llm_error(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    mock_client.messages.create = AsyncMock(side_effect=Exception("API rate limit"))
    adapter._client = mock_client

    with pytest.raises(LLMError, match="Anthropic API error"):
        await adapter.generate(_req("hello"))


async def test_empty_system_not_passed(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """Empty system string should not be sent as API parameter."""
    adapter._client = mock_client

    await adapter.generate(_req("hello", system=""))

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs


async def test_usage_populated_from_response(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """Usage is surfaced via LLMResponse.usage (no _last_usage side channel)."""
    text_block = SimpleNamespace(text="response")
    response = SimpleNamespace(
        content=[text_block],
        usage=_mock_usage(
            input_tokens=200,
            output_tokens=80,
            cache_read_input_tokens=1500,
            cache_creation_input_tokens=0,
        ),
    )
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    result = await adapter.generate(_req("hello", system="personality text"))

    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 80
    assert result.usage.cache_read_tokens == 1500
    assert result.usage.cache_write_tokens == 0


async def test_cache_write_on_first_call(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """First call with system surfaces cache_creation_input_tokens."""
    text_block = SimpleNamespace(text="ok")
    response = SimpleNamespace(
        content=[text_block],
        usage=_mock_usage(cache_creation_input_tokens=1024),
    )
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    result = await adapter.generate(_req("hi", system="system text"))

    assert result.usage.cache_write_tokens == 1024


async def test_generate_empty_response_raises(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """Empty content list raises LLMError."""
    response = SimpleNamespace(content=[], usage=_mock_usage())
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    with pytest.raises(LLMError, match="Empty response"):
        await adapter.generate(_req("hello"))


async def test_generate_no_text_block_raises(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    """Response block without .text raises LLMError."""
    bad_block = SimpleNamespace(type="image")
    response = SimpleNamespace(content=[bad_block], usage=_mock_usage())
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    with pytest.raises(LLMError, match="Unexpected response block"):
        await adapter.generate(_req("hello"))


async def test_generate_with_temperature(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    adapter._client = mock_client
    await adapter.generate(_req("hello", temperature=0.5))

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.5


# --- generate_with_tools ---


async def test_generate_with_tools_returns_tool_response(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="search", input={"q": "hi"}
    )
    response = SimpleNamespace(
        content=[tool_block],
        stop_reason="tool_use",
        usage=_mock_usage(),
    )
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    tools = [
        ToolDefinition(
            name="search",
            description="Search",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    result = await adapter.generate_with_tools(
        _tool_req([{"role": "user", "content": "find something"}], tools)
    )
    assert isinstance(result, LLMToolResponse)
    assert result.stop_reason == "tool_use"
    assert len(result.content_blocks) == 1
    assert result.model == "claude-sonnet-4-6"
    assert result.usage.input_tokens == 100
    mock_client.messages.create.assert_awaited_once()


async def test_generate_with_tools_system_and_temp(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    text_block = SimpleNamespace(text="I'll search for that")
    response = SimpleNamespace(
        content=[text_block],
        stop_reason="end_turn",
        usage=_mock_usage(),
    )
    mock_client.messages.create = AsyncMock(return_value=response)
    adapter._client = mock_client

    tools = [
        ToolDefinition(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    await adapter.generate_with_tools(
        _tool_req(
            [{"role": "user", "content": "hi"}],
            tools,
            system="Be helpful",
            temperature=0.3,
        )
    )
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.3
    assert "system" in call_kwargs


async def test_generate_with_tools_error(
    adapter: AnthropicAPIAdapter, mock_client: AsyncMock
) -> None:
    mock_client.messages.create = AsyncMock(side_effect=Exception("timeout"))
    adapter._client = mock_client

    tools = [
        ToolDefinition(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    with pytest.raises(LLMError, match="tool_use error"):
        await adapter.generate_with_tools(
            _tool_req([{"role": "user", "content": "hi"}], tools)
        )


async def test_lazy_client_init(adapter: AnthropicAPIAdapter) -> None:
    """Client is created lazily on first generate call."""
    assert adapter._client is None

    mock_anthropic = MagicMock()
    text_block = SimpleNamespace(text="ok")
    response = SimpleNamespace(content=[text_block], usage=_mock_usage())
    mock_instance = AsyncMock()
    mock_instance.messages.create = AsyncMock(return_value=response)
    mock_anthropic.AsyncAnthropic.return_value = mock_instance

    with (
        patch.dict(
            "os.environ",
            {
                "BOT_TOKEN": "fake",
                "CHANNEL_ID": "@t",
                "ADMIN_USER_ID": "1",
                "ANTHROPIC_API_KEY": "fake_anthropic_test_key",
            },
        ),
        patch("src.llm.anthropic_api.get_settings") as mock_settings,
        patch.dict("sys.modules", {"anthropic": mock_anthropic}),
    ):
        mock_settings.return_value = SimpleNamespace(
            anthropic_api_key="fake_anthropic_test_key"
        )
        adapter._client = None
        result = await adapter.generate(_req("test"))

    assert result.text == "ok"
