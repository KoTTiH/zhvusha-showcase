from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    from pathlib import Path

from src.skills.base import AgentContext
from src.skills.channel_writer.skill import ChannelWriterSkill
from src.skills.kwork_monitor.skill import KworkMonitorSkill
from src.skills.workspace_session.skill import WorkspaceSessionSkill


def _context(mode: str = "personal") -> AgentContext:
    return AgentContext(
        user_id=12345,
        chat_id=12345,
        mode=mode,  # type: ignore[arg-type]
        message_id=1,
        bot=AsyncMock(),
    )


async def test_kwork_rejects_assistant():
    """KworkMonitorSkill is a BackgroundSkill: execute returns empty failure."""
    skill = KworkMonitorSkill()
    result = await skill.execute("/kwork", _context(mode="assistant"))
    assert result.success is False


async def test_channel_writer_rejects_social(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    result = await skill.execute("/post hello", _context(mode="social"))
    assert result.success is False


async def test_workspace_session_rejects_assistant():
    skill = WorkspaceSessionSkill()
    result = await skill.execute("/morning", _context(mode="assistant"))
    assert result.success is False
