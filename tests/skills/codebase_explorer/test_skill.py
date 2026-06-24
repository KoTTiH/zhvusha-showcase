"""Codebase explorer routing tests for ordinary personal chat."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock


def _context(**overrides: Any):
    from src.skills.base import AgentContext

    defaults = {
        "user_id": 1,
        "chat_id": 2,
        "mode": "personal",
        "message_id": 10,
        "bot": None,
    }
    return AgentContext(**{**defaults, **overrides})


async def test_codebase_explorer_handles_project_source_compare_request(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    runner = AsyncMock(return_value="проверенный ответ")
    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=runner,
    )

    score = await skill.can_handle(
        "Можешь изучить проект и сравнить с постом выше?",
        _context(),
    )
    result = await skill.execute(
        "Можешь изучить проект и сравнить с постом выше?",
        _context(),
    )

    assert score == 0.86
    assert result.response == "проверенный ответ"
    assert "Недавний чат" in runner.await_args.kwargs["user_prompt"]


async def test_codebase_explorer_starts_background_when_bot_is_available(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    class Bot:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_message(
            self,
            *,
            chat_id: int,
            text: str,
            parse_mode: str | None = None,
        ) -> None:
            del chat_id, parse_mode
            self.messages.append(text)

    class BackgroundRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def start_background(self, **kwargs: Any) -> Any:
            self.calls += 1
            await kwargs["completion_callback"]("готовый background report")
            return object()

    bot = Bot()
    runner = BackgroundRunner()
    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=AsyncMock(return_value="sync should not run"),
        background_runner=runner,
    )

    result = await skill.execute(
        "Можешь изучить проект и сравнить с постом выше?",
        _context(bot=bot),
    )

    assert runner.calls == 1
    assert "взяла в фоновую agent-задачу" in result.response
    assert bot.messages == ["готовый background report"]


async def test_codebase_explorer_starts_background_for_vscode_without_bot(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    class BackgroundRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def start_background(self, **kwargs: Any) -> Any:
            self.calls += 1
            await kwargs["completion_callback"]("готовый background report")
            return SimpleNamespace(id="job-vscode-1")

    runner = BackgroundRunner()
    sync_runner = AsyncMock(return_value="sync should not run")
    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=sync_runner,
        background_runner=runner,
    )

    result = await skill.execute(
        "Проверь проект как CTO",
        _context(
            metadata={
                "interface": "vscode",
                "digital_scenario_id": "ai_cto_projects",
            },
        ),
    )

    assert runner.calls == 1
    sync_runner.assert_not_awaited()
    assert "фоновую read-only agent-задачу" in result.response
    assert result.metadata["agent_job_id"] == "job-vscode-1"
    assert result.metadata["agent_job_result_pending"] is True
    assert result.metadata["agent_job_status"] == "running"


async def test_codebase_explorer_does_not_claim_codex_decision_only_handoff(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=AsyncMock(),
    )

    score = await skill.can_handle(
        "\n".join(
            [
                "Codex/operator handoff, sender=codex, не Никита.",
                "operator_handoff_mode: decision_only_existing_agent_evidence",
                "Agent Runtime Job Evidence:",
                "- job_id=job-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa status=done",
                "Handoff prompt: Не запускай новый source_compare job.",
            ]
        ),
        _context(metadata={"interface": "vscode"}),
    )

    assert score == 0.0


async def test_codebase_explorer_waits_for_promised_material(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    class BackgroundRunner:
        def __init__(self) -> None:
            self.awaiting_calls = 0
            self.started = 0

        async def create_awaiting_input(self, **kwargs: Any) -> Any:
            self.awaiting_calls += 1
            assert "сравнить с постом" in kwargs["user_prompt"]
            return object()

        async def start_background(self, **kwargs: Any) -> Any:
            del kwargs
            self.started += 1
            return object()

    runner = BackgroundRunner()
    sync_runner = AsyncMock(return_value="sync should not run")
    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=sync_runner,
        background_runner=runner,
    )

    result = await skill.execute(
        "Можешь изучить проект и сравнить с постом? Щас скину.",
        _context(),
    )

    assert runner.awaiting_calls == 1
    assert runner.started == 0
    sync_runner.assert_not_awaited()
    assert "Кидай материал" in result.response


async def test_codebase_explorer_does_not_steal_plain_post_explanation(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=AsyncMock(),
    )

    score = await skill.can_handle("Объясни пост, ща скину", _context())

    assert score == 0.0


async def test_codebase_explorer_does_not_treat_direct_address_as_project_marker(
    tmp_path: Path,
) -> None:
    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill

    skill = CodebaseExplorerSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        explorer_runner=AsyncMock(),
    )

    score = await skill.can_handle(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        _context(),
    )

    assert score == 0.0
