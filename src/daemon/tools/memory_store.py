"""Store an episodic memory entry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.daemon.tools.base import DaemonTool, ToolResult

if TYPE_CHECKING:
    from src.memory import EpisodicMemoryProtocol as EpisodicMemory


class MemoryStoreTool(DaemonTool):
    """Record an episode to episodic memory."""

    name = "memory_store"
    description = "Записать эпизод в эпизодическую память"
    requires_approval = False

    def __init__(self, episodic: EpisodicMemory) -> None:
        self._episodic = episodic

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Record episode. Params: content (str), source (str), importance (float)."""
        content = params.get("content", "")
        if not content:
            return ToolResult(success=False, message="Empty content")

        try:
            episode_id = await self._episodic.record(
                content=content,
                user_id=0,
                chat_type="personal",
                role="assistant",
                source=params.get("source", "daemon"),
                importance=params.get("importance", 0.5),
            )
            return ToolResult(
                success=True,
                message=f"Episode #{episode_id} recorded",
                data={"episode_id": episode_id},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Memory error: {e}")
