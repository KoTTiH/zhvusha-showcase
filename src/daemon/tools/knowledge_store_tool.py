"""Add knowledge to the knowledge base."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.daemon.tools.base import DaemonTool, ToolResult

if TYPE_CHECKING:
    from src.knowledge import KnowledgeStore


class KnowledgeStoreTool(DaemonTool):
    """Add an entry to the knowledge base."""

    name = "knowledge_store"
    description = "Сохранить запись в базу знаний"
    # Approval happens at staging level (propose_change), not tool level
    requires_approval = False

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Stage entry for review. Params: title (str), content (str), tags (list[str], optional)."""
        title = params.get("title", "")
        content = params.get("content", "")
        if not title or not content:
            return ToolResult(success=False, message="Title and content required")

        tags: Any = params.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        try:
            staging_id = await self._store.propose_change(
                operation="add",
                target_entry_id=None,
                proposed_changes={
                    "title": title,
                    "content": content,
                    "tags": tags,
                    "source": params.get("source", "daemon"),
                },
                reason=params.get("reason", "Daemon proposed knowledge entry"),
                proposed_by="daemon",
            )
            return ToolResult(
                success=True,
                message=f"Staged #{staging_id} for review: {title}",
                data={"staging_id": staging_id},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Staging error: {e}")
