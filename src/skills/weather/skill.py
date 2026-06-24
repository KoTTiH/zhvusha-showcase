"""WeatherSkill — trivial InlineSkill returning hard-coded temperature.

Dry-run target for Phase 15 spec lifecycle calibration.
Real API integration deferred to a separate spec.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from src.skills.base import (
    AgentContext,
    InlineSkill,
    SideEffect,
    SkillResult,
)


class WeatherSkill(InlineSkill):
    """Return hard-coded weather for a given city. No real API calls."""

    name: ClassVar[str] = "weather"
    description: ClassVar[str] = "Hard-coded weather stub for /weather <city>"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    triggers: ClassVar[list[str]] = ["/weather"]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = []
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(self, *, admin_user_id: int) -> None:
        self._admin_user_id = admin_user_id

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Match ``/weather <city>`` for admin in personal mode."""
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip().lower()
        if text == "/weather" or text.startswith("/weather "):
            return 1.0
        return 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Return hard-coded temperature for the requested city."""
        del context
        city = message.strip().removeprefix("/weather").strip()
        if not city:
            return SkillResult(
                success=True,
                response="Укажи город: `/weather <city>`",
            )
        return SkillResult(
            success=True,
            response=f"Погода в {city}: 12.5°C",
        )
