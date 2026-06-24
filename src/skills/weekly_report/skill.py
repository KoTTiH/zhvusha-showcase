"""Inline skill that renders the weekly three-pillar report."""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol

from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult
from src.skills.weekly_report.formatter import (
    WeeklyReportSnapshot,
    format_weekly_report,
)

_TRIGGER = "/weekly_report"


class WeeklyReportProvider(Protocol):
    async def build_snapshot(self, *, days: int = 7) -> WeeklyReportSnapshot: ...


class WeeklyReportSkill(InlineSkill):
    """Render a concise report of movement across the three pillars."""

    name: ClassVar[str] = "weekly_report"
    description: ClassVar[str] = "Builds weekly report across priority pillars"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [_TRIGGER]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.READS_FILESYSTEM]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(self, *, admin_user_id: int, report_provider: WeeklyReportProvider):
        self._admin_user_id = admin_user_id
        self._provider = report_provider

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip().lower()
        return 1.0 if text == _TRIGGER or text.startswith(_TRIGGER + " ") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        days = _parse_days(message)
        return await self.run_once(days=days)

    async def run_once(self, *, days: int = 7) -> SkillResult:
        snapshot = await self._provider.build_snapshot(days=days)
        return SkillResult(
            success=True,
            response=format_weekly_report(snapshot),
            metadata={
                "days": days,
                "topics": len(snapshot.topics),
                "drafts": snapshot.generated_drafts,
            },
        )


def _parse_days(message: str) -> int:
    parts = message.split()
    if len(parts) < 2:
        return 7
    try:
        return max(1, min(int(parts[1]), 30))
    except ValueError:
        return 7
