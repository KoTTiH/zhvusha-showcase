"""Tests for ``src.llm.codex_cli.CodexCLIAdapter``."""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from src.llm.protocols import LLMError, LLMRequest, LLMToolRequest, ToolDefinition


class _FakeProcess:
    def __init__(self, *, returncode: int, output_path: Path, final: str = "ok"):
        self.returncode = returncode
        self._output_path = output_path
        self._final = final
        self.stdin = b""

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        self.stdin = data
        if self._final:
            self._output_path.write_text(self._final, encoding="utf-8")
        return b"stdout", b"stderr"


class _CancelledProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.waited = False

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        del data
        raise asyncio.CancelledError

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


def _req(
    prompt: str = "Привет",
    *,
    system: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        system=system,
        model=model,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        caller="chat",
    )


def _tool_req(output: str = "верни ответ") -> LLMToolRequest:
    return LLMToolRequest(
        messages=[{"role": "user", "content": output}],
        tools=[
            ToolDefinition(
                name="search_knowledge",
                description="Искать в базе знаний",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        system="Будь Жвушей",
        model="gpt-5.5",
        caller="chat_agentic",
    )


def _patch_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    final: str = "Ответ Codex",
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_process_factory(*args: str, **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        output_flag = args.index("--output-last-message")
        output_path = Path(args[output_flag + 1])
        process = _FakeProcess(
            returncode=returncode,
            output_path=output_path,
            final=final,
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(
        "src.llm.codex_cli._create_process",
        fake_process_factory,
    )
    return captured


class TestCodexCLIAdapter:
    async def test_generate_returns_last_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(monkeypatch, final="Живой ответ")
        result = await CodexCLIAdapter(codex_path="codex").generate(
            _req("как дела?", system="Будь собой")
        )

        assert result.text == "Живой ответ"
        assert result.model == "default"
        args = captured["args"]
        assert args[:3] == ("codex", "--ask-for-approval", "never")
        assert "exec" in args
        assert "--ephemeral" in args
        assert "--skip-git-repo-check" in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert "--model" not in args
        instruction_values = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
        assert "instructions=" in instruction_values[-1]
        encoded_system = instruction_values[-1].removeprefix("instructions=")
        assert tomllib.loads(f"value = {encoded_system}")["value"] == "Будь собой"
        prompt = captured["process"].stdin.decode("utf-8")
        assert prompt == "как дела?"

    async def test_generate_passes_custom_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(monkeypatch)
        result = await CodexCLIAdapter(codex_path="codex-test").generate(
            _req("test", model="gpt-5.5")
        )

        assert result.model == "gpt-5.5"
        args = captured["args"]
        assert args[0] == "codex-test"
        assert args[args.index("--ask-for-approval") + 1] == "never"
        assert args[args.index("--model") + 1] == "gpt-5.5"

    async def test_generate_passes_reasoning_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(monkeypatch)
        await CodexCLIAdapter().generate(
            _req("test", model="gpt-5.5", reasoning_effort="xhigh")
        )

        args = captured["args"]
        assert args[args.index("--model") + 1] == "gpt-5.5"
        assert args[args.index("-c") + 1] == 'model_reasoning_effort="xhigh"'

    async def test_generate_escapes_multiline_system_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(monkeypatch)
        system = 'Первая строка\n"вторая" строка'
        await CodexCLIAdapter().generate(_req("user text", system=system))

        args = captured["args"]
        instruction_values = [
            args[i + 1]
            for i, arg in enumerate(args)
            if arg == "-c" and args[i + 1].startswith("instructions=")
        ]
        assert len(instruction_values) == 1
        encoded_system = instruction_values[0].removeprefix("instructions=")
        assert tomllib.loads(f"value = {encoded_system}")["value"] == system

    async def test_generate_with_tools_returns_final_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(
            monkeypatch,
            final='{"type":"final","text":"готовый ответ"}',
        )

        result = await CodexCLIAdapter().generate_with_tools(_tool_req())

        assert result.stop_reason == "end_turn"
        assert result.content_blocks[0].text == "готовый ответ"
        prompt = captured["process"].stdin.decode("utf-8")
        assert "search_knowledge" in prompt
        assert "верни ответ" in prompt

    async def test_generate_with_tools_returns_tool_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        _patch_process(
            monkeypatch,
            final=(
                '{"type":"tool_calls","calls":[{"id":"call_1",'
                '"name":"search_knowledge","input":{"query":"голос Жвуши"}}]}'
            ),
        )

        result = await CodexCLIAdapter().generate_with_tools(_tool_req())

        block = result.content_blocks[0]
        assert result.stop_reason == "tool_use"
        assert block.type == "tool_use"
        assert block.id == "call_1"
        assert block.name == "search_knowledge"
        assert block.input == {"query": "голос Жвуши"}

    async def test_generate_with_tools_plain_text_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        _patch_process(monkeypatch, final="обычный ответ без json")

        result = await CodexCLIAdapter().generate_with_tools(_tool_req())

        assert result.stop_reason == "end_turn"
        assert result.content_blocks[0].text == "обычный ответ без json"

    async def test_generate_with_tools_ignores_tool_calls_when_no_tools_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        raw = '{"type":"tool_calls","calls":[{"name":"search_knowledge","input":{}}]}'
        _patch_process(monkeypatch, final=raw)
        request = LLMToolRequest(
            messages=[{"role": "user", "content": "финализируй"}],
            tools=[],
            system="Будь Жвушей",
            model="gpt-5.5",
        )

        result = await CodexCLIAdapter().generate_with_tools(request)

        assert result.stop_reason == "end_turn"
        assert result.content_blocks[0].text == raw

    async def test_generate_strips_openai_api_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        captured = _patch_process(monkeypatch)
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "fake_openai_secret",
                "OPENAI_BASE_URL": "https://api.openai.example",
                "BOT_TOKEN": "fake",
            },
            clear=True,
        ):
            await CodexCLIAdapter().generate(_req("test"))

        env = captured["env"]
        assert "OPENAI_API_KEY" not in env
        assert "OPENAI_BASE_URL" not in env
        assert env["BOT_TOKEN"] == "fake"

    async def test_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        _patch_process(monkeypatch, returncode=2, final="")

        with pytest.raises(LLMError, match="Codex CLI exited"):
            await CodexCLIAdapter().generate(_req("test"))

    async def test_cancellation_terminates_codex_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.codex_cli import CodexCLIAdapter

        process = _CancelledProcess()

        async def fake_process_factory(*args: str, **kwargs: Any) -> _CancelledProcess:
            del args, kwargs
            return process

        monkeypatch.setattr(
            "src.llm.codex_cli._create_process",
            fake_process_factory,
        )

        with pytest.raises(asyncio.CancelledError):
            await CodexCLIAdapter().generate(_req("test"))

        assert process.terminated is True
        assert process.killed is False
        assert process.waited is True
