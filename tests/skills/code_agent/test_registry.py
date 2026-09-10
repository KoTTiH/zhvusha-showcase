"""Contract tests for the Codex-only self-coding backend layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.contract
_PROJECT_ROOT = Path("project-root")


class _Backend:
    def __init__(self, name: str, result: Any = None, error: Exception | None = None):
        self.name = name
        self.run_architect = AsyncMock(side_effect=error, return_value=result)
        self.run_explorer = AsyncMock(side_effect=error, return_value=result)
        self.run_editor = AsyncMock(side_effect=error, return_value=result)


def _architect_request() -> Any:
    from src.skills.code_agent.protocols import ArchitectRequest

    return ArchitectRequest(
        system_prompt="shared rules",
        user_prompt="draft spec",
        cwd=_PROJECT_ROOT,
        model="",
    )


def _editor_request() -> Any:
    from src.skills.code_agent.protocols import EditorRequest

    return EditorRequest(
        system_prompt="shared editor rules",
        user_prompt="make test green",
        cwd=_PROJECT_ROOT,
        project_root=_PROJECT_ROOT,
        whitelist_paths=["src/skills/weather/skill.py"],
        existing_tests_to_update_paths=[],
        model="",
    )


def _explorer_request() -> Any:
    from src.skills.code_agent.protocols import ExplorerRequest

    return ExplorerRequest(
        system_prompt="shared explorer rules",
        user_prompt="inspect before discussion",
        cwd=_PROJECT_ROOT,
        model="",
    )


class TestCodeAgentRegistry:
    async def test_primary_codex_is_used_when_available(self) -> None:
        from src.skills.code_agent.protocols import CodeAgentBackend, CodeAgentResult
        from src.skills.code_agent.registry import CodeAgentRegistry

        codex = _Backend("codex_cli", result=CodeAgentResult("codex", "codex_cli"))
        registry = CodeAgentRegistry(
            backends={"codex_cli": cast("CodeAgentBackend", codex)},
            backend="codex_cli",
        )

        result = await registry.run_architect(_architect_request())

        assert result.backend == "codex_cli"
        codex.run_architect.assert_awaited_once()

    async def test_unknown_backend_fails_without_fallback(self) -> None:
        from src.skills.code_agent.protocols import CodeAgentUnavailableError
        from src.skills.code_agent.registry import CodeAgentRegistry

        registry = CodeAgentRegistry(backends={}, backend="missing_backend")

        with pytest.raises(CodeAgentUnavailableError, match="missing_backend"):
            await registry.run_architect(_architect_request())

    async def test_editor_receives_shared_request_rules(self) -> None:
        from src.skills.code_agent.protocols import CodeAgentBackend, CodeAgentResult
        from src.skills.code_agent.registry import CodeAgentRegistry

        codex = _Backend("codex_cli", result=CodeAgentResult("codex", "codex_cli"))
        registry = CodeAgentRegistry(
            backends={"codex_cli": cast("CodeAgentBackend", codex)},
            backend="codex_cli",
        )
        request = _editor_request()

        result = await registry.run_editor(request)

        assert result.backend == "codex_cli"
        codex.run_editor.assert_awaited_once_with(request)

    async def test_explorer_receives_shared_request_rules(self) -> None:
        from src.skills.code_agent.protocols import CodeAgentBackend, CodeAgentResult
        from src.skills.code_agent.registry import CodeAgentRegistry

        codex = _Backend("codex_cli", result=CodeAgentResult("codex", "codex_cli"))
        registry = CodeAgentRegistry(
            backends={"codex_cli": cast("CodeAgentBackend", codex)},
            backend="codex_cli",
        )
        request = _explorer_request()

        result = await registry.run_explorer(request)

        assert result.backend == "codex_cli"
        codex.run_explorer.assert_awaited_once_with(request)

    def test_claude_backend_name_is_rejected_for_self_coding(self) -> None:
        from src.skills.code_agent.registry import CodeAgentRegistry

        with pytest.raises(ValueError, match=r"Claude.*self-coding"):
            CodeAgentRegistry(backends={}, backend="claude_agent_sdk")

    def test_backend_order_is_codex_only(self) -> None:
        from src.skills.code_agent.registry import CodeAgentRegistry

        registry = CodeAgentRegistry(
            backends={},
            backend="codex_cli",
        )

        assert registry.backend_order == ("codex_cli",)

    def test_build_codex_registry_passes_reasoning_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.code_agent import registry as registry_module

        captured = {}

        class FakeCodexBackend:
            def __init__(
                self,
                *,
                codex_path: str,
                model: str,
                reasoning_effort: str,
                timeout_seconds: float,
            ) -> None:
                captured["codex_path"] = codex_path
                captured["model"] = model
                captured["reasoning_effort"] = reasoning_effort
                captured["timeout_seconds"] = timeout_seconds

        monkeypatch.setattr(registry_module, "CodexCliBackend", FakeCodexBackend)

        registry_module.build_codex_registry(
            backend="codex_cli",
            codex_path="codex-test",
            codex_model="gpt-5.5",
            reasoning_effort="xhigh",
            timeout_seconds=7200.0,
        )

        assert captured == {
            "codex_path": "codex-test",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "timeout_seconds": 7200.0,
        }

    def test_self_coding_runners_do_not_import_claude_sdk(self) -> None:
        paths = [
            Path("src/skills/ideation_to_spec/sdk_runner.py"),
            Path("src/skills/implement_spec/sdk_runner.py"),
            Path("src/skills/delegate/skill.py"),
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert "claude_agent_sdk" not in text
