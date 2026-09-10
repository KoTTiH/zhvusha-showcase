"""Contract tests for the Codex CLI code-agent backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract
_PROJECT_ROOT = Path("project-root")


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int,
        output_path: Path,
        final: str = "done",
        stdout: bytes = b"stdout",
        stderr: bytes = b"stderr",
    ):
        self.returncode = returncode
        self._output_path = output_path
        self._final = final
        self._stdout = stdout
        self._stderr = stderr
        self.stdin = b""

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        self.stdin = data
        if self._final:
            self._output_path.write_text(self._final, encoding="utf-8")
        return self._stdout, self._stderr


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdin = b""
        self.terminated = False

    async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
        self.stdin = data
        await asyncio.sleep(10)
        return b"", b""

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    async def wait(self) -> int | None:
        return self.returncode


class _FakePipeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _StreamingProcess:
    def __init__(
        self,
        *,
        returncode: int,
        output_path: Path,
        stdout: asyncio.StreamReader,
        stderr: asyncio.StreamReader,
        final: str = "done",
    ) -> None:
        self.returncode = returncode
        self._output_path = output_path
        self._final = final
        self.stdin = _FakePipeStdin()
        self.stdout = stdout
        self.stderr = stderr

    async def wait(self) -> int:
        if self._final:
            self._output_path.write_text(self._final, encoding="utf-8")
        return self.returncode


def _patch_process_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    final: str = "done",
    stdout: bytes = b"stdout",
    stderr: bytes = b"stderr",
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
            stdout=stdout,
            stderr=stderr,
        )
        captured["process"] = process
        return process

    monkeypatch.setattr(
        "src.skills.code_agent.codex_cli._create_process",
        fake_process_factory,
    )
    return captured


class TestCodexCliBackend:
    async def test_architect_runs_codex_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ArchitectRequest

        captured = _patch_process_factory(monkeypatch, final="architect text")
        backend = CodexCliBackend(codex_path="codex-test", model="gpt-test")
        result = await backend.run_architect(
            ArchitectRequest(
                system_prompt="SYSTEM RULES",
                user_prompt="USER REQUEST",
                cwd=_PROJECT_ROOT,
                model="",
            )
        )

        assert result.text == "architect text"
        assert result.backend == "codex_cli"
        args = captured["args"]
        assert args[:3] == ("codex-test", "--ask-for-approval", "never")
        assert args[args.index("--model") + 1] == "gpt-test"
        assert args.index("--model") < args.index("exec")
        assert args[args.index("--cd") + 1] == "project-root"
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert "--skip-git-repo-check" in args
        assert "--ephemeral" in args
        assert "--json" not in args
        prompt = captured["process"].stdin.decode("utf-8")
        assert "SYSTEM RULES" in prompt
        assert "USER REQUEST" in prompt

    async def test_architect_defaults_to_medium_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ArchitectRequest

        captured = _patch_process_factory(monkeypatch, final="architect text")
        backend = CodexCliBackend(
            codex_path="codex-test",
            model="gpt-test",
            reasoning_effort="xhigh",
        )

        await backend.run_architect(
            ArchitectRequest(
                system_prompt="SYSTEM RULES",
                user_prompt="USER REQUEST",
                cwd=_PROJECT_ROOT,
                model="",
            )
        )

        args = captured["args"]
        assert args[args.index("-c") + 1] == 'model_reasoning_effort="medium"'

    async def test_architect_honors_explicit_reasoning_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ArchitectRequest

        captured = _patch_process_factory(monkeypatch, final="architect text")
        backend = CodexCliBackend(
            codex_path="codex-test",
            model="gpt-test",
            reasoning_effort="xhigh",
        )

        await backend.run_architect(
            ArchitectRequest(
                system_prompt="SYSTEM RULES",
                user_prompt="USER REQUEST",
                cwd=_PROJECT_ROOT,
                model="",
                reasoning_effort="high",
            )
        )

        args = captured["args"]
        assert args[args.index("-c") + 1] == 'model_reasoning_effort="high"'

    async def test_persistent_explorer_records_thread_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ExplorerRequest

        captured = _patch_process_factory(
            monkeypatch,
            final="explorer text",
            stdout=(
                b'{"type":"thread.started",'
                b'"thread_id":"019e1cf5-a63c-7ca1-a44e-44e555239799"}\n'
            ),
        )

        result = await CodexCliBackend(codex_path="codex-test").run_explorer(
            ExplorerRequest(
                system_prompt="EXPLORER RULES",
                user_prompt="READ",
                cwd=_PROJECT_ROOT,
                persist_session=True,
            )
        )

        args = captured["args"]
        assert "--json" in args
        assert "--ephemeral" not in args
        assert "resume" not in args
        assert result.session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"

    async def test_resume_uses_existing_thread_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ExplorerRequest

        captured = _patch_process_factory(
            monkeypatch,
            final="continued",
            stdout=(
                b'{"type":"thread.started",'
                b'"thread_id":"019e1cf5-a63c-7ca1-a44e-44e555239799"}\n'
            ),
        )

        result = await CodexCliBackend(codex_path="codex-test").run_explorer(
            ExplorerRequest(
                system_prompt="EXPLORER RULES",
                user_prompt="CONTINUE",
                cwd=_PROJECT_ROOT,
                session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
            )
        )

        args = captured["args"]
        assert args[:5] == (
            "codex-test",
            "--ask-for-approval",
            "never",
            "exec",
            "resume",
        )
        assert args[5] == "019e1cf5-a63c-7ca1-a44e-44e555239799"
        assert "--cd" not in args
        assert "--sandbox" not in args
        assert "--json" in args
        assert "--ephemeral" not in args
        assert result.session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"

    async def test_resume_keeps_existing_thread_id_when_json_event_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ExplorerRequest

        _patch_process_factory(
            monkeypatch,
            final="continued",
            stdout=b"plain output without thread event\n",
        )

        result = await CodexCliBackend(codex_path="codex-test").run_explorer(
            ExplorerRequest(
                system_prompt="EXPLORER RULES",
                user_prompt="CONTINUE",
                cwd=_PROJECT_ROOT,
                session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
            )
        )

        assert result.session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"

    async def test_editor_runs_codex_with_shared_whitelist_rules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import EditorRequest

        captured = _patch_process_factory(monkeypatch, final="editor text")
        backend = CodexCliBackend(codex_path="codex-test")
        result = await backend.run_editor(
            EditorRequest(
                system_prompt="EDITOR RULES",
                user_prompt="MAKE GREEN",
                cwd=_PROJECT_ROOT,
                project_root=_PROJECT_ROOT,
                whitelist_paths=["src/skills/weather/skill.py"],
                existing_tests_to_update_paths=[
                    "tests/skills/weather/test_contract.py"
                ],
                model="",
            )
        )

        assert result.backend == "codex_cli"
        args = captured["args"]
        assert args[args.index("--sandbox") + 1] == "workspace-write"
        assert args[args.index("--ask-for-approval") + 1] == "never"
        prompt = captured["process"].stdin.decode("utf-8")
        assert "EDITOR RULES" in prompt
        assert "src/skills/weather/skill.py" in prompt
        assert "tests/skills/weather/test_contract.py" in prompt

    async def test_explorer_runs_codex_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ExplorerRequest

        captured = _patch_process_factory(monkeypatch, final="explorer text")

        backend = CodexCliBackend(codex_path="codex-test", model="gpt-test")
        result = await backend.run_explorer(
            ExplorerRequest(
                system_prompt="EXPLORER RULES",
                user_prompt="READ AND EXPLAIN",
                cwd=_PROJECT_ROOT,
                model="",
            )
        )

        assert result.text == "explorer text"
        assert result.backend == "codex_cli"
        args = captured["args"]
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("--ask-for-approval") + 1] == "never"
        prompt = captured["process"].stdin.decode("utf-8")
        assert "EXPLORER RULES" in prompt
        assert "READ AND EXPLAIN" in prompt

    async def test_progress_reader_handles_long_codex_trace_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ExplorerRequest

        stdout = asyncio.StreamReader()
        stderr = asyncio.StreamReader()
        stdout.feed_data(
            b'{"type":"trace","payload":"'
            + (b"x" * 70000)
            + b'"}\nTG_STATUS: \xd0\xa1\xd0\xbe\xd0\xb1\xd0\xb8\xd1\x80\xd0\xb0\xd1\x8e evidence.\n'
        )
        stdout.feed_eof()
        stderr.feed_eof()
        captured: dict[str, Any] = {}

        async def fake_process_factory(*args: str, **kwargs: Any) -> _StreamingProcess:
            del kwargs
            captured["args"] = args
            output_flag = args.index("--output-last-message")
            output_path = Path(args[output_flag + 1])
            process = _StreamingProcess(
                returncode=0,
                output_path=output_path,
                stdout=stdout,
                stderr=stderr,
                final="explorer final",
            )
            captured["process"] = process
            return process

        monkeypatch.setattr(
            "src.skills.code_agent.codex_cli._create_process",
            fake_process_factory,
        )
        progress: list[str] = []

        async def progress_callback(message: str) -> None:
            progress.append(message)

        result = await CodexCliBackend(codex_path="codex-test").run_explorer(
            ExplorerRequest(
                system_prompt="EXPLORER RULES",
                user_prompt="READ AND EXPLAIN",
                cwd=_PROJECT_ROOT,
                progress_callback=progress_callback,
            )
        )

        assert result.text == "explorer final"
        assert progress == ["Собираю evidence."]
        assert b"READ AND EXPLAIN" in captured["process"].stdin.data

    async def test_strips_openai_api_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import patch

        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ArchitectRequest

        captured = _patch_process_factory(monkeypatch)
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "fake_openai_secret",
                "OPENAI_BASE_URL": "https://api.openai.example",
                "BOT_TOKEN": "fake",
            },
            clear=True,
        ):
            await CodexCliBackend(codex_path="codex-test").run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )

        env = captured["env"]
        assert "OPENAI_API_KEY" not in env
        assert "OPENAI_BASE_URL" not in env
        assert env["BOT_TOKEN"] == "fake"

    async def test_delegate_passes_reasoning_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import DelegateRequest

        captured = _patch_process_factory(monkeypatch, final="delegate text")
        backend = CodexCliBackend(
            codex_path="codex-test",
            model="gpt-5.5",
            reasoning_effort="xhigh",
        )
        result = await backend.run_delegate(
            DelegateRequest(task="RUN MORNING", cwd=_PROJECT_ROOT)
        )

        assert result.text == "delegate text"
        args = captured["args"]
        assert args[args.index("--model") + 1] == "gpt-5.5"
        assert args[args.index("-c") + 1] == 'model_reasoning_effort="xhigh"'
        assert args.index("-c") < args.index("exec")
        prompt = captured["process"].stdin.decode("utf-8")
        assert "No-Downgrade" in prompt
        assert "RUN MORNING" in prompt

    async def test_missing_codex_binary_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import (
            ArchitectRequest,
            CodeAgentUnavailableError,
        )

        async def missing_binary(*args: str, **kwargs: Any) -> Any:
            del args, kwargs
            raise FileNotFoundError("missing")

        monkeypatch.setattr(
            "src.skills.code_agent.codex_cli._create_process",
            missing_binary,
        )

        with pytest.raises(CodeAgentUnavailableError, match="codex_cli"):
            await CodexCliBackend(codex_path="missing-codex").run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )

    async def test_nonzero_exit_is_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import (
            ArchitectRequest,
            CodeAgentExecutionError,
        )

        _patch_process_factory(monkeypatch, returncode=2, final="")

        with pytest.raises(CodeAgentExecutionError, match="codex_cli"):
            await CodexCliBackend(codex_path="codex-test").run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )

    async def test_nonzero_exit_reports_tail_instead_of_prompt_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import (
            ArchitectRequest,
            CodeAgentExecutionError,
        )

        stderr = (
            "OpenAI Codex v0.129.0\n"
            "# System Rules\n"
            "very long prompt head\n"
            + "\n".join(f"noise {idx}" for idx in range(60))
            + "\nactual failure: hook denied edit outside whitelist\n"
        ).encode()
        _patch_process_factory(monkeypatch, returncode=2, final="", stderr=stderr)

        with pytest.raises(CodeAgentExecutionError) as exc_info:
            await CodexCliBackend(codex_path="codex-test").run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )

        message = str(exc_info.value)
        assert "actual failure: hook denied edit outside whitelist" in message
        assert "very long prompt head" not in message

    async def test_times_out_and_stops_hung_codex_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import (
            ArchitectRequest,
            CodeAgentExecutionError,
        )

        process = _HangingProcess()

        async def hanging_process_factory(*args: str, **kwargs: Any) -> _HangingProcess:
            del args, kwargs
            return process

        monkeypatch.setattr(
            "src.skills.code_agent.codex_cli._create_process",
            hanging_process_factory,
        )

        with pytest.raises(CodeAgentExecutionError, match="timed out"):
            await CodexCliBackend(
                codex_path="codex-test",
                timeout_seconds=0.01,
            ).run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )

        assert process.terminated is True

    async def test_cancellation_stops_hung_codex_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.codex_cli import CodexCliBackend
        from src.skills.code_agent.protocols import ArchitectRequest

        process = _HangingProcess()

        async def hanging_process_factory(*args: str, **kwargs: Any) -> _HangingProcess:
            del args, kwargs
            return process

        monkeypatch.setattr(
            "src.skills.code_agent.codex_cli._create_process",
            hanging_process_factory,
        )

        task = asyncio.create_task(
            CodexCliBackend(codex_path="codex-test").run_architect(
                ArchitectRequest(
                    system_prompt="x",
                    user_prompt="y",
                    cwd=_PROJECT_ROOT,
                    model="",
                )
            )
        )
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.terminated is True


class TestCodexStatusExtraction:
    def test_extracts_only_explicit_telegram_status(self) -> None:
        from src.skills.code_agent.codex_cli import extract_codex_session_status

        assert (
            extract_codex_session_status("TG_STATUS: Сейчас дочитаю контекст.")
            == "Сейчас дочитаю контекст."
        )
        assert (
            extract_codex_session_status("• TG_STATUS: Добавляю RED-тесты.")
            == "Добавляю RED-тесты."
        )

    def test_ignores_raw_tool_trace_and_working_lines(self) -> None:
        from src.skills.code_agent.codex_cli import extract_codex_session_status

        assert extract_codex_session_status("• Explored") is None
        assert extract_codex_session_status("  └ Read test_skill.py") is None
        assert extract_codex_session_status("• Ran wc -l src/x.py") is None
        assert (
            extract_codex_session_status("Working (2m 16s • esc to interrupt") is None
        )
