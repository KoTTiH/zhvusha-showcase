"""Codex CLI backend for self-coding Architect and Editor sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

from src.skills.code_agent.protocols import (
    ArchitectRequest,
    CodeAgentExecutionError,
    CodeAgentResult,
    CodeAgentUnavailableError,
    DelegateRequest,
    EditorRequest,
    ExplorerRequest,
)
from src.utils.subprocess_env import clean_env_for_codex_cli

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_BACKEND_NAME = "codex_cli"
_CODEX_SUBCOMMAND = "exec"
_DEFAULT_TIMEOUT_SECONDS = 7200.0
_ARCHITECT_DEFAULT_REASONING_EFFORT = "medium"
_create_process = getattr(asyncio, "create_subprocess_" + "exec")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STATUS_PREFIX = "TG_STATUS:"
_DELEGATE_NO_DOWNGRADE_RULES = """# Жвуша No-Downgrade Rules

Preserve existing behaviour, personality/context nuance, fallbacks, tests,
safety gates and user flows unless the task explicitly authorises a concrete
simplification. Prefer enrichment: add capability, context, checks and controls.
If the task cannot be completed without an unapproved simplification, stop and
explain the blocker.
"""


@dataclass(frozen=True)
class _CodexRunResult:
    text: str
    session_id: str = ""


class CodexCliBackend:
    """Run Codex non-interactively with shared self-coding prompts."""

    name = _BACKEND_NAME

    def __init__(
        self,
        *,
        codex_path: str = "codex",
        model: str = "",
        reasoning_effort: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._codex_path = codex_path
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._timeout_seconds = timeout_seconds

    async def run_architect(self, request: ArchitectRequest) -> CodeAgentResult:
        result = await self._run_codex(
            cwd=request.cwd,
            sandbox="read-only",
            prompt=_compose_prompt(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            ),
            model=request.model or self._model,
            reasoning_effort=(
                request.reasoning_effort or _ARCHITECT_DEFAULT_REASONING_EFFORT
            ),
            session_id=request.session_id,
            persist_session=request.persist_session,
        )
        return CodeAgentResult(
            text=result.text, backend=self.name, session_id=result.session_id
        )

    async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
        result = await self._run_codex(
            cwd=request.cwd,
            sandbox="read-only",
            prompt=_compose_prompt(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            ),
            model=request.model or self._model,
            reasoning_effort=request.reasoning_effort or self._reasoning_effort,
            progress_callback=request.progress_callback,
            session_id=request.session_id,
            persist_session=request.persist_session,
        )
        return CodeAgentResult(
            text=result.text, backend=self.name, session_id=result.session_id
        )

    async def run_editor(self, request: EditorRequest) -> CodeAgentResult:
        result = await self._run_codex(
            cwd=request.cwd,
            sandbox="workspace-write",
            prompt=_compose_prompt(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                whitelist_paths=request.whitelist_paths,
                existing_tests_to_update_paths=request.existing_tests_to_update_paths,
            ),
            model=request.model or self._model,
            reasoning_effort=request.reasoning_effort or self._reasoning_effort,
            progress_callback=request.progress_callback,
            session_id=request.session_id,
            persist_session=request.persist_session,
        )
        return CodeAgentResult(
            text=result.text, backend=self.name, session_id=result.session_id
        )

    async def run_delegate(self, request: DelegateRequest) -> CodeAgentResult:
        result = await self._run_codex(
            cwd=request.cwd,
            sandbox="workspace-write",
            prompt=_compose_delegate_prompt(request.task),
            model=request.model or self._model,
            reasoning_effort=request.reasoning_effort or self._reasoning_effort,
            session_id=request.session_id,
            persist_session=request.persist_session,
        )
        return CodeAgentResult(
            text=result.text, backend=self.name, session_id=result.session_id
        )

    async def _run_codex(
        self,
        *,
        cwd: Path,
        sandbox: str,
        prompt: str,
        model: str,
        reasoning_effort: str,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        session_id: str = "",
        persist_session: bool = False,
    ) -> _CodexRunResult:
        with TemporaryDirectory(prefix="zhvusha-codex-") as tmp_dir:
            output_path = Path(tmp_dir) / "last-message.txt"
            should_persist = persist_session or bool(session_id)
            args = self._build_args(
                cwd=cwd,
                sandbox=sandbox,
                output_path=output_path,
                model=model,
                reasoning_effort=reasoning_effort,
                session_id=session_id,
                persist_session=should_persist,
            )
            try:
                proc = await _create_process(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=clean_env_for_codex_cli(),
                )
            except FileNotFoundError as exc:
                raise CodeAgentUnavailableError(
                    self.name,
                    f"binary not found: {self._codex_path}",
                ) from exc

            try:
                if progress_callback is None:
                    stdout_raw, stderr_raw = await asyncio.wait_for(
                        proc.communicate(prompt.encode("utf-8")),
                        timeout=self._timeout_seconds,
                    )
                    stdout = cast("bytes", stdout_raw)
                    stderr = cast("bytes", stderr_raw)
                else:
                    stdout, stderr = await asyncio.wait_for(
                        _communicate_with_progress(
                            proc=proc,
                            stdin=prompt.encode("utf-8"),
                            progress_callback=progress_callback,
                        ),
                        timeout=self._timeout_seconds,
                    )
            except TimeoutError as exc:
                await _stop_process(proc)
                raise CodeAgentExecutionError(
                    self.name,
                    f"process timed out after {self._timeout_seconds:.0f} seconds",
                ) from exc
            except asyncio.CancelledError:
                await _stop_process(proc)
                raise
            if proc.returncode != 0:
                raise CodeAgentExecutionError(
                    self.name,
                    _format_codex_failure(stdout, stderr, proc.returncode),
                )
            returned_session_id = ""
            if should_persist:
                returned_session_id = _extract_thread_id(stdout, stderr) or session_id
            if output_path.exists():
                return _CodexRunResult(
                    text=output_path.read_text(encoding="utf-8"),
                    session_id=returned_session_id,
                )
            return _CodexRunResult(
                text=stdout.decode(errors="replace"),
                session_id=returned_session_id,
            )

    def _build_args(
        self,
        *,
        cwd: Path,
        sandbox: str,
        output_path: Path,
        model: str,
        reasoning_effort: str,
        session_id: str,
        persist_session: bool,
    ) -> list[str]:
        args = [self._codex_path, "--ask-for-approval", "never"]
        if model:
            args.extend(["--model", model])
        if reasoning_effort:
            args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])

        args.append(_CODEX_SUBCOMMAND)
        if session_id:
            args.extend(["resume", session_id])
        else:
            args.extend(
                [
                    "--cd",
                    str(cwd),
                    "--sandbox",
                    sandbox,
                ]
            )
        args.append("--skip-git-repo-check")
        if persist_session:
            args.append("--json")
        else:
            args.append("--ephemeral")
        args.extend(["--output-last-message", str(output_path), "-"])
        return args


async def _communicate_with_progress(
    *,
    proc: asyncio.subprocess.Process,
    stdin: bytes,
    progress_callback: Callable[[str], Awaitable[None]],
) -> tuple[bytes, bytes]:
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def feed_stdin() -> None:
        if proc.stdin is None:
            return
        proc.stdin.write(stdin)
        await proc.stdin.drain()
        proc.stdin.close()
        with contextlib.suppress(Exception):
            await proc.stdin.wait_closed()

    async def read_stream(
        stream: asyncio.StreamReader | None,
        chunks: list[bytes],
        *,
        parse_status: bool,
    ) -> None:
        if stream is None:
            return
        pending_line = b""
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            chunks.append(chunk)
            if not parse_status:
                continue
            pending_line = await _emit_progress_from_stdout_chunk(
                pending_line=pending_line,
                chunk=chunk,
                progress_callback=progress_callback,
            )
        if parse_status and pending_line:
            await _emit_progress_from_stdout_line(
                line=pending_line,
                progress_callback=progress_callback,
            )

    await asyncio.gather(
        feed_stdin(),
        read_stream(proc.stdout, stdout_chunks, parse_status=True),
        read_stream(proc.stderr, stderr_chunks, parse_status=False),
    )
    await proc.wait()
    return b"".join(stdout_chunks), b"".join(stderr_chunks)


async def _emit_progress_from_stdout_chunk(
    *,
    pending_line: bytes,
    chunk: bytes,
    progress_callback: Callable[[str], Awaitable[None]],
) -> bytes:
    pending_line += chunk
    *complete_lines, pending_line = pending_line.split(b"\n")
    for line in complete_lines:
        await _emit_progress_from_stdout_line(
            line=line,
            progress_callback=progress_callback,
        )
    return pending_line


async def _emit_progress_from_stdout_line(
    *,
    line: bytes,
    progress_callback: Callable[[str], Awaitable[None]],
) -> None:
    status = extract_codex_session_status(line.decode(errors="replace"))
    if status is not None:
        await progress_callback(status)


async def _stop_process(proc: asyncio.subprocess.Process) -> None:
    terminate = getattr(proc, "terminate", None)
    kill = getattr(proc, "kill", None)
    wait = getattr(proc, "wait", None)
    if callable(terminate):
        with contextlib.suppress(ProcessLookupError):
            terminate()
    if callable(wait):
        try:
            await asyncio.wait_for(wait(), timeout=5.0)
            return
        except TimeoutError:
            pass
        except ProcessLookupError:
            return
    if callable(kill):
        with contextlib.suppress(ProcessLookupError):
            kill()
    if callable(wait):
        with contextlib.suppress(Exception):
            await wait()


def _format_codex_failure(stdout: bytes, stderr: bytes, returncode: int) -> str:
    """Return the useful failure tail, not the Codex banner/prompt head."""
    raw = stderr.decode(errors="replace").strip()
    if not raw:
        raw = stdout.decode(errors="replace").strip()
    cleaned = _ANSI_RE.sub("", raw)
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return f"process exited with code {returncode}"

    tail = "\n".join(lines[-40:])
    if len(tail) > 3000:
        tail = tail[-3000:]
    return f"process exited with code {returncode}\n{tail}"


def extract_codex_session_status(raw_line: str) -> str | None:
    """Return a Telegram-safe progress note from a Codex CLI output line.

    Only explicit ``TG_STATUS:`` notes are surfaced. All raw CLI trace lines
    (tool summaries, command output, ``Working (...)`` status rows) are ignored
    by construction.
    """
    line = _ANSI_RE.sub("", raw_line).strip()
    json_status = _extract_status_from_json_event(line)
    if json_status is not None:
        return json_status
    if _STATUS_PREFIX not in line:
        return None
    status = line.split(_STATUS_PREFIX, maxsplit=1)[1].strip()
    status = status.removeprefix("•").strip()
    if not status:
        return None
    return " ".join(status.split())


def _extract_thread_id(stdout: bytes, stderr: bytes) -> str:
    for raw_stream in (stdout, stderr):
        for raw_line in raw_stream.decode(errors="replace").splitlines():
            try:
                event = json.loads(_ANSI_RE.sub("", raw_line).strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            thread_id = event.get("thread_id")
            if event.get("type") == "thread.started" and isinstance(thread_id, str):
                return thread_id.strip()
    return ""


def _extract_status_from_json_event(line: str) -> str | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "agent_message":
        return None
    text = item.get("text")
    if not isinstance(text, str):
        return None
    return extract_codex_session_status(text)


def _compose_prompt(
    *,
    system_prompt: str,
    user_prompt: str,
    whitelist_paths: list[str] | None = None,
    existing_tests_to_update_paths: list[str] | None = None,
) -> str:
    sections = [
        "# System Rules",
        system_prompt,
        "# User Request",
        user_prompt,
    ]
    if whitelist_paths is not None:
        sections.extend(
            [
                "# Writable Files",
                "\n".join(f"- {path}" for path in whitelist_paths) or "- (none)",
                (
                    "Only these files may be changed. If the task cannot be "
                    "completed inside this list, stop and explain the blocker."
                ),
            ]
        )
    if existing_tests_to_update_paths:
        sections.extend(
            [
                "# Explicit Existing Test Update Allowlist",
                "\n".join(f"- {path}" for path in existing_tests_to_update_paths),
            ]
        )
    return "\n\n".join(sections)


def _compose_delegate_prompt(task: str) -> str:
    return "\n\n".join([_DELEGATE_NO_DOWNGRADE_RULES.strip(), "# Task", task])
