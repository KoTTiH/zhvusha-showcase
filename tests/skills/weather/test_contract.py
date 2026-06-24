"""Contract tests for WeatherSkill."""

from __future__ import annotations

import pytest
from src.skills.base import AgentContext, SkillResult
from src.skills.weather.skill import WeatherSkill


@pytest.fixture()
def skill() -> WeatherSkill:
    return WeatherSkill(admin_user_id=12345)


@pytest.fixture()
def ctx() -> AgentContext:
    return AgentContext(user_id=12345, chat_id=12345, mode="personal")


@pytest.mark.asyncio()
async def test_returns_hardcoded_temp(skill: WeatherSkill, ctx: AgentContext) -> None:
    result = await skill.execute("/weather Madrid", ctx)
    assert isinstance(result, SkillResult)
    assert result.success is True
    assert "12.5" in result.response
    assert "Madrid" in result.response


@pytest.mark.asyncio()
async def test_can_handle_weather_command(
    skill: WeatherSkill, ctx: AgentContext
) -> None:
    confidence = await skill.can_handle("/weather Berlin", ctx)
    assert confidence == 1.0


@pytest.mark.asyncio()
async def test_rejects_non_admin(skill: WeatherSkill) -> None:
    ctx = AgentContext(user_id=99999, chat_id=99999, mode="personal")
    confidence = await skill.can_handle("/weather Berlin", ctx)
    assert confidence == 0.0


@pytest.mark.asyncio()
async def test_rejects_non_personal_mode(skill: WeatherSkill) -> None:
    ctx = AgentContext(user_id=12345, chat_id=12345, mode="assistant")
    confidence = await skill.can_handle("/weather Berlin", ctx)
    assert confidence == 0.0
