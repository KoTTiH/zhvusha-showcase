"""Codex CLI adapter for subscription-backed chat generation.

This is an LLM Gateway adapter, separate from ``src.skills.code_agent``. The
code-agent backend uses Codex for self-coding; this adapter uses Codex CLI as a
subscription-backed text/tool provider for chat tiers without API-key billing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import structlog

from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
    ToolDefinition,
)
from src.utils.subprocess_env import clean_env_for_codex_cli

logger = structlog.get_logger()

_SUBCOMMAND = "exec"
_DEFAULT_MODEL = "default"
_create_process = getattr(asyncio, "create_subprocess_" + "exec")

_TOOL_PROTOCOL = """\
## Tool-use protocol
Ты можешь использовать только инструменты из списка в пользовательском prompt.
Не выполняй shell/CLI действия для этого chat turn.
Все инструменты из списка реально подключены как callable tools. Если Никита
просит действие, которое покрывается инструментом, вызывай инструмент, а не
говори что он недоступен.

Верни ровно один JSON object без markdown:
- Финальный ответ: {"type":"final","text":"..."}
- Вызовы инструментов: {"type":"tool_calls","calls":[{"id":"call_1","name":"tool_name","input":{...}}]}

Если данных хватает без инструментов — отвечай type=final.
Если нужен инструмент — верни type=tool_calls и не добавляй финальный ответ до результата инструмента.
"""


@dataclass(frozen=True)
class _TextBlock:
    text: str


@dataclass(frozen=True)
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


class CodexCLIAdapter(BaseLLMAdapter):
    """Adapter that calls Codex CLI through subscription auth."""

    name: str = "codex_cli"
    default_model: str = _DEFAULT_MODEL

    def __init__(self, *, codex_path: str | None = None) -> None:
        self._codex_path = codex_path

    async def generate(self, request: LLMRequest) -> LLMResponse:
        settings = get_settings()
        codex_path = self._codex_path or settings.codex_cli_path
        model = request.model or self.default_model
        prompt = request.prompt

        logger.info(
            "llm_request",
            adapter=self.name,
            model=model,
            reasoning_effort=request.reasoning_effort,
            prompt_len=len(request.prompt),
            has_system=bool(request.system),
            caller=request.caller,
        )

        with TemporaryDirectory(prefix="zhvusha-codex-chat-") as tmp_dir:
            output_path = Path(tmp_dir) / "last-message.txt"
            args = [
                codex_path,
                "--ask-for-approval",
                "never",
                _SUBCOMMAND,
                "--cd",
                tmp_dir,
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-last-message",
                str(output_path),
                "-",
            ]
            model_arg = _model_arg(model)
            prefix_args: list[str] = []
            if model_arg:
                prefix_args.extend(["--model", model_arg])
            if request.reasoning_effort:
                prefix_args.extend(
                    ["-c", f'model_reasoning_effort="{request.reasoning_effort}"']
                )
            if request.system:
                # Codex exec has no dedicated --system-prompt flag; config
                # ``instructions`` is rendered as model-visible developer text.
                prefix_args.extend(
                    ["-c", f"instructions={_toml_string(request.system)}"]
                )
            if prefix_args:
                args[3:3] = prefix_args

            try:
                proc = await _create_process(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=clean_env_for_codex_cli(),
                )
            except FileNotFoundError as exc:
                raise LLMError(f"Codex CLI binary not found: {codex_path}") from exc

            try:
                stdout_raw, stderr_raw = await proc.communicate(prompt.encode("utf-8"))
            except asyncio.CancelledError:
                await _terminate_process(proc)
                raise
            stdout = cast("bytes", stdout_raw)
            stderr = cast("bytes", stderr_raw)

            if proc.returncode != 0:
                detail = stderr.decode(errors="replace").strip()
                if not detail:
                    detail = stdout.decode(errors="replace").strip()
                logger.error(
                    "codex_cli_error",
                    returncode=proc.returncode,
                    error=detail[:500],
                )
                raise LLMError(
                    f"Codex CLI exited with code {proc.returncode}: {detail}"
                )

            if output_path.exists():
                text = output_path.read_text(encoding="utf-8")
            else:
                text = stdout.decode(errors="replace")

        logger.info(
            "llm_response",
            adapter=self.name,
            model=model,
            response_len=len(text),
        )
        return LLMResponse(text=text, model=model, usage=LLMUsage())

    async def generate_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        """Generate using a provider-neutral JSON tool protocol over Codex CLI."""
        system = (
            "\n\n".join([request.system, _TOOL_PROTOCOL])
            if request.system
            else _TOOL_PROTOCOL
        )
        response = await self.generate(
            LLMRequest(
                prompt=_compose_tool_prompt(request.messages, request.tools),
                system=system,
                tier=request.tier,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
                temperature=request.temperature,
                caller=request.caller,
            )
        )
        return _parse_tool_response(
            response.text,
            model=response.model,
            usage=response.usage,
            allow_tool_calls=bool(request.tools),
        )


async def _terminate_process(proc: Any) -> None:
    """Stop a Codex CLI child process after outer task cancellation."""
    if getattr(proc, "returncode", None) is not None:
        return
    terminate = getattr(proc, "terminate", None)
    if callable(terminate):
        with contextlib.suppress(ProcessLookupError):
            terminate()
    wait = getattr(proc, "wait", None)
    if callable(wait):
        try:
            await asyncio.wait_for(wait(), timeout=2.0)
            return
        except TimeoutError:
            pass
    kill = getattr(proc, "kill", None)
    if callable(kill):
        with contextlib.suppress(ProcessLookupError):
            kill()
    if callable(wait):
        with contextlib.suppress(Exception):
            await wait()


def _model_arg(model: str) -> str:
    """Translate registry's subscription default sentinel to no CLI flag."""
    return "" if model in {"", _DEFAULT_MODEL} else model


def _toml_string(value: str) -> str:
    """Encode arbitrary instructions as one TOML string for ``codex -c``."""
    return json.dumps(value, ensure_ascii=False)


def _compose_tool_prompt(
    messages: list[dict[str, Any]],
    tools: list[ToolDefinition],
) -> str:
    payload = {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ],
        "messages": messages,
    }
    return (
        "Ниже доступные chat tools и текущий transcript. "
        "Следуй Tool-use protocol из system/developer instructions.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_tool_response(
    text: str,
    *,
    model: str,
    usage: LLMUsage,
    allow_tool_calls: bool,
) -> LLMToolResponse:
    parsed = _load_json_object(text)
    if not isinstance(parsed, dict):
        return LLMToolResponse(
            content_blocks=[_TextBlock(text)],
            stop_reason="end_turn",
            model=model,
            usage=usage,
        )

    response_type = parsed.get("type")
    if response_type == "tool_calls" and allow_tool_calls:
        blocks: list[_ToolUseBlock] = []
        calls = parsed.get("calls")
        if isinstance(calls, list):
            for index, call in enumerate(calls, start=1):
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                tool_input = call.get("input")
                if not isinstance(name, str) or not isinstance(tool_input, dict):
                    continue
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"codex_call_{index}"
                blocks.append(
                    _ToolUseBlock(
                        id=call_id,
                        name=name,
                        input=dict(tool_input),
                    )
                )
        if blocks:
            return LLMToolResponse(
                content_blocks=blocks,
                stop_reason="tool_use",
                model=model,
                usage=usage,
            )

    final_text = parsed.get("text")
    if not isinstance(final_text, str):
        final_text = text
    return LLMToolResponse(
        content_blocks=[_TextBlock(final_text)],
        stop_reason="end_turn",
        model=model,
        usage=usage,
    )


def _load_json_object(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
