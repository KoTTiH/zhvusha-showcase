import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from src.core.config import get_settings
from src.llm.claude_cli import ClaudeCLIAdapter
from src.llm.protocols import LLMError, LLMRequest, LLMResponse


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def adapter():
    return ClaudeCLIAdapter()


def _req(
    prompt: str = "test", *, system: str = "", model: str | None = None
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system=system,
        model=model,
        tier="strategist",
        caller="test",
    )


def _make_process(stdout: bytes, returncode: int = 0, stderr: bytes = b""):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


def _ok_response(text: str = "Generated text") -> bytes:
    return json.dumps({"type": "result", "is_error": False, "result": text}).encode()


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_returns_response(adapter):
    proc = _make_process(_ok_response("Hello world"))

    with patch("src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc):
        result = await adapter.generate(_req("say hello"))

    assert isinstance(result, LLMResponse)
    assert result.text == "Hello world"
    assert result.model == "sonnet"
    # Claude CLI reports zero usage — tracked coarsely via record_cli_call
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_passes_model(adapter):
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test", model="opus"))

    args = mock_exec.call_args[0]
    assert "--model" in args
    model_idx = args.index("--model")
    assert args[model_idx + 1] == "opus"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_passes_system_prompt(adapter):
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test", system="Be brief"))

    args = mock_exec.call_args[0]
    assert "--system-prompt" in args
    idx = args.index("--system-prompt")
    assert args[idx + 1] == "Be brief"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_no_system_prompt(adapter):
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test"))

    args = mock_exec.call_args[0]
    assert "--system-prompt" not in args


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_uses_default_model(adapter):
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test"))

    args = mock_exec.call_args[0]
    model_idx = args.index("--model")
    assert args[model_idx + 1] == "sonnet"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_nonzero_exit_raises(adapter):
    proc = _make_process(b"", returncode=1, stderr=b"auth failed")

    with (
        patch("src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc),
        pytest.raises(LLMError, match="exited with code 1"),
    ):
        await adapter.generate(_req("test"))


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_invalid_json_raises(adapter):
    proc = _make_process(b"not json at all")

    with (
        patch("src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc),
        pytest.raises(LLMError, match="parse"),
    ):
        await adapter.generate(_req("test"))


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
    },
    clear=True,
)
async def test_generate_is_error_flag_raises(adapter):
    data = json.dumps({"is_error": True, "result": "rate limited"}).encode()
    proc = _make_process(data)

    with (
        patch("src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc),
        pytest.raises(LLMError, match="rate limited"),
    ):
        await adapter.generate(_req("test"))


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "/custom/path/claude",
    },
    clear=True,
)
async def test_generate_uses_custom_cli_path(adapter):
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test"))

    args = mock_exec.call_args[0]
    assert args[0] == "/custom/path/claude"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "CHANNEL_ID": "@test",
        "ADMIN_USER_ID": "1",
        "CLAUDE_CLI_PATH": "claude",
        "ANTHROPIC_API_KEY": "fake_anthropic_secret",
    },
    clear=True,
)
async def test_generate_strips_anthropic_api_key(adapter):
    """ANTHROPIC_API_KEY must not leak into subprocess env."""
    proc = _make_process(_ok_response())

    with patch(
        "src.llm.claude_cli.asyncio.create_subprocess_exec", return_value=proc
    ) as mock_exec:
        await adapter.generate(_req("test"))

    env = mock_exec.call_args.kwargs["env"]
    assert "ANTHROPIC_API_KEY" not in env
