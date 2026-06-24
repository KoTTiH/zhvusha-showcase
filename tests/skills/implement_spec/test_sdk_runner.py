"""Contract tests for Codex-backed self-coding runner wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

_FAKE_CWD = Path("/var/empty/zhvusha-fake-cwd")


class _FakeCodexBackend:
    def __init__(self, *, codex_path: str = "codex", model: str = "") -> None:
        self.codex_path = codex_path
        self.model = model

    async def run_editor(self, request: Any) -> Any:
        from src.skills.code_agent.protocols import CodeAgentResult

        self.request = request
        return CodeAgentResult(text="editor ok", backend="codex_cli")

    async def run_architect(self, request: Any) -> Any:
        from src.skills.code_agent.protocols import CodeAgentResult

        self.request = request
        return CodeAgentResult(text="architect ok", backend="codex_cli")


class TestEditorRunner:
    async def test_passes_shared_editor_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.implement_spec import sdk_runner

        created: list[_FakeCodexBackend] = []

        def factory(*, codex_path: str = "codex", model: str = "") -> _FakeCodexBackend:
            backend = _FakeCodexBackend(codex_path=codex_path, model=model)
            created.append(backend)
            return backend

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", factory)

        result = await sdk_runner.run_editor_sdk(
            user_prompt="do thing",
            system_prompt="you are editor",
            cwd=_FAKE_CWD,
            project_root=_FAKE_CWD,
            whitelist_paths=["src/skills/weather/skill.py"],
            existing_tests_to_update_paths=[
                "tests/skills/weather/test_contract.py",
            ],
            model="gpt-test",
            codex_path="codex-test",
        )

        assert result.text == "editor ok"
        assert result.backend == "codex_cli"
        backend = created[0]
        assert backend.codex_path == "codex-test"
        assert backend.model == "gpt-test"
        assert backend.request.user_prompt == "do thing"
        assert backend.request.system_prompt == "you are editor"
        assert backend.request.cwd == _FAKE_CWD
        assert backend.request.project_root == _FAKE_CWD
        assert backend.request.whitelist_paths == ["src/skills/weather/skill.py"]
        assert backend.request.existing_tests_to_update_paths == [
            "tests/skills/weather/test_contract.py"
        ]
        assert backend.request.progress_callback is None
        assert backend.request.session_id == ""
        assert backend.request.persist_session is False

    async def test_passes_editor_session_controls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.implement_spec import sdk_runner

        created: list[_FakeCodexBackend] = []

        def factory(*, codex_path: str = "codex", model: str = "") -> _FakeCodexBackend:
            backend = _FakeCodexBackend(codex_path=codex_path, model=model)
            created.append(backend)
            return backend

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", factory)

        await sdk_runner.run_editor_sdk(
            user_prompt="do thing",
            system_prompt="you are editor",
            cwd=_FAKE_CWD,
            project_root=_FAKE_CWD,
            whitelist_paths=["src/skills/weather/skill.py"],
            session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
            persist_session=True,
        )

        request = created[0].request
        assert request.session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"
        assert request.persist_session is True

    async def test_passes_progress_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.implement_spec import sdk_runner

        created: list[_FakeCodexBackend] = []

        def factory(*, codex_path: str = "codex", model: str = "") -> _FakeCodexBackend:
            backend = _FakeCodexBackend(codex_path=codex_path, model=model)
            created.append(backend)
            return backend

        async def progress_callback(message: str) -> None:
            del message

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", factory)

        await sdk_runner.run_editor_sdk(
            user_prompt="do thing",
            system_prompt="you are editor",
            cwd=_FAKE_CWD,
            project_root=_FAKE_CWD,
            whitelist_paths=["src/skills/weather/skill.py"],
            progress_callback=progress_callback,
        )

        assert created[0].request.progress_callback is progress_callback

    async def test_unavailable_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent.protocols import CodeAgentUnavailableError
        from src.skills.implement_spec import sdk_runner

        class FailingBackend(_FakeCodexBackend):
            async def run_editor(self, request: Any) -> Any:
                del request
                raise CodeAgentUnavailableError("codex_cli", "missing")

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", FailingBackend)

        with pytest.raises(sdk_runner.SDKUnavailableError):
            await sdk_runner.run_editor_sdk(
                user_prompt="x",
                system_prompt="x",
                cwd=_FAKE_CWD,
                project_root=_FAKE_CWD,
                whitelist_paths=["src/skills/weather/skill.py"],
            )


class TestArchitectRunner:
    async def test_returns_architect_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.ideation_to_spec import sdk_runner

        created: list[_FakeCodexBackend] = []

        def factory(*, codex_path: str = "codex", model: str = "") -> _FakeCodexBackend:
            backend = _FakeCodexBackend(codex_path=codex_path, model=model)
            created.append(backend)
            return backend

        monkeypatch.setattr(sdk_runner, "CodexCliBackend", factory)

        out = await sdk_runner.run_architect_sdk(
            user_prompt="draft",
            system_prompt="rules",
            cwd=_FAKE_CWD,
            model="gpt-test",
            codex_path="codex-test",
        )

        assert out == "architect ok"
        backend = created[0]
        assert backend.codex_path == "codex-test"
        assert backend.model == "gpt-test"
        assert backend.request.user_prompt == "draft"
        assert backend.request.system_prompt == "rules"
        assert backend.request.cwd == _FAKE_CWD
