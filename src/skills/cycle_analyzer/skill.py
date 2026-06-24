"""Inline lookup over archived self-coding cycle insights."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult

if TYPE_CHECKING:
    from src.archive.store import ArchiveStore

_TRIGGER = "/archive_lookup"


class CycleAnalyzerSkill(InlineSkill):
    """Expose deterministic archive lookup to Никита."""

    name: ClassVar[str] = "cycle_analyzer"
    description: ClassVar[str] = "Lookup archived self-coding cycle insights"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [_TRIGGER]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.READS_FILESYSTEM]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self, *, admin_user_id: int, archive_store: ArchiveStore | None
    ) -> None:
        self._admin_user_id = admin_user_id
        self._store = archive_store

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip().lower()
        return 1.0 if text == _TRIGGER or text.startswith(_TRIGGER + " ") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        query = message.strip()[len(_TRIGGER) :].strip()
        if not query:
            return SkillResult(
                success=True,
                response="Укажи запрос: `/archive_lookup <тема или slug>`.",
            )
        if self._store is None:
            return SkillResult(
                success=False,
                response="Archive store не подключен: нужен DATABASE_URL и alembic upgrade.",
            )
        nodes = await self._store.lookup(query, top_k=5)
        if not nodes:
            return SkillResult(
                success=True, response="В archive_nodes ничего не найдено."
            )
        lines = ["**Archive lookup**"]
        for node in nodes:
            lines.append(
                f"  • `{node.slug}` · {node.status.value} · "
                f"{node.commit_sha[:12] if node.commit_sha else 'no commit'}\n"
                f"    {node.insight[:220]}"
            )
        return SkillResult(
            success=True,
            response="\n".join(lines),
            metadata={"count": len(nodes)},
        )
