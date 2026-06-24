"""Background skill that prepares a ranked morning digest."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from src.skills.base import AgentContext, BackgroundSkill, SideEffect, SkillResult
from src.skills.morning_digest.formatter import format_morning_digest

if TYPE_CHECKING:
    from src.skills.morning_digest.formatter import DigestTopic


class MorningDigestProvider(Protocol):
    """Narrow provider interface for testable digest generation."""

    async def list_topics(self, *, limit: int = 20) -> list[DigestTopic]: ...


class MorningDigestSkill(BackgroundSkill):
    """Prepare morning message from ranked topic backlog."""

    name: ClassVar[str] = "morning_digest"
    description: ClassVar[str] = "Формирует утренний digest по topic backlog"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    trigger_type: ClassVar[Literal["cron", "event", "interval"]] = "cron"
    trigger_config: ClassVar[dict[str, object]] = {
        "hour": 8,
        "timezone": "Europe/Moscow",
    }
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.READS_FILESYSTEM]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(self, *, topic_provider: MorningDigestProvider) -> None:
        self._topics = topic_provider

    async def run_once(self, *, limit: int = 20) -> SkillResult:
        topics = await self._topics.list_topics(limit=limit)
        return SkillResult(success=True, response=format_morning_digest(topics))

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        return await self.run_once()
