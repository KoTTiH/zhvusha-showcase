"""Bot command surface for isolated external skill smoke checks."""

from __future__ import annotations

from pathlib import Path


async def test_external_skill_smoke_command_is_admin_only(tmp_path: Path) -> None:
    from src.bot.main import _external_skill_smoke_reply
    from src.skills.base import AgentContext

    reply = await _external_skill_smoke_reply(
        "/external_skill_smoke",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        workspace_root=tmp_path,
    )

    assert reply == "Эта команда доступна только Никите."


async def test_external_skill_smoke_command_runs_without_touching_registry(
    tmp_path: Path,
) -> None:
    from src.bot.main import _external_skill_smoke_reply
    from src.skills.base import AgentContext

    workspace = tmp_path / "workspace"
    reply = await _external_skill_smoke_reply(
        "/external_skill_smoke",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        workspace_root=workspace,
    )

    assert reply is not None
    assert "External Skill Smoke status" in reply
    assert "PASS" in reply
    assert "approval-" not in reply
    assert (workspace / "skills" / "external" / "smoke" / "latest").exists()
    assert not (workspace / "skills" / "external" / "registry").exists()
