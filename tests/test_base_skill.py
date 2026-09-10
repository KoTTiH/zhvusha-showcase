"""Tests for the v4 ``BaseSkill`` hierarchy directly.

Per-skill contract and integration tests live under ``tests/skills/``; this
file exercises the abstract hierarchy itself (BaseSkill abstractness,
InlineSkill/BackgroundSkill/DelegatedSkill default method behaviour, frozen
dataclasses).
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import ClassVar, Literal

import pytest
from src.skills.base import (
    AgentContext,
    BackgroundSkill,
    BaseSkill,
    DelegatedSkill,
    ExecutionPlan,
    Feedback,
    InlineSkill,
    SideEffect,
    SimulatedResult,
    SkillResult,
)


class _ConcreteInline(InlineSkill):
    name: ClassVar[str] = "test_inline"
    description: ClassVar[str] = "Inline test"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if "test" in message else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        return SkillResult(success=True, response=f"inline-{message}")


class _ConcreteBackground(BackgroundSkill):
    name: ClassVar[str] = "test_bg"
    description: ClassVar[str] = "Background test"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "analyst"

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        return SkillResult(success=True, response="bg-tick")


class _ConcreteDelegated(DelegatedSkill):
    name: ClassVar[str] = "test_delegated"
    description: ClassVar[str] = "Delegated test"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"
    executor: ClassVar[str] = "test_executor"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/do") else 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="delegated",
            human_summary=f"delegate: {message}",
            estimated_tokens=1000,
            estimated_cost_usd=Decimal("0.05"),
            estimated_duration_seconds=10.0,
            side_effects_invoked=list(self.side_effects),
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        return SkillResult(success=True, response=f"delegated-{message}")


def _ctx() -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal")


def test_cannot_instantiate_abstract_base_skill() -> None:
    with pytest.raises(TypeError):
        BaseSkill()  # type: ignore[abstract]


def test_agent_context_is_frozen() -> None:
    ctx = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.user_id = 2  # type: ignore[misc]


def test_skill_result_is_frozen() -> None:
    result = SkillResult(success=True, response="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False  # type: ignore[misc]


def test_execution_plan_is_frozen() -> None:
    plan = ExecutionPlan(
        skill_name="x",
        skill_type="inline",
        human_summary="summary",
        estimated_tokens=1,
        estimated_cost_usd=Decimal("0.001"),
        estimated_duration_seconds=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.skill_name = "y"  # type: ignore[misc]


def test_feedback_is_frozen() -> None:
    fb = Feedback(skill_name="x", user_id=1, rating="positive")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fb.user_id = 2  # type: ignore[misc]


def test_simulated_result_is_frozen() -> None:
    sim = SimulatedResult(
        would_succeed=True,
        would_produce="result",
        dependencies_available=True,
        estimated_actual_cost=Decimal("0.01"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        sim.would_succeed = False  # type: ignore[misc]


class TestInlineSkill:
    async def test_can_handle(self) -> None:
        skill = _ConcreteInline()
        assert await skill.can_handle("test please", _ctx()) == 1.0
        assert await skill.can_handle("no match", _ctx()) == 0.0

    async def test_execute(self) -> None:
        skill = _ConcreteInline()
        result = await skill.execute("msg", _ctx())
        assert result.success is True
        assert result.response == "inline-msg"

    async def test_default_prepare(self) -> None:
        skill = _ConcreteInline()
        plan = await skill.prepare("msg", _ctx())
        assert plan.skill_name == "test_inline"
        assert plan.skill_type == "inline"

    async def test_default_dry_run(self) -> None:
        skill = _ConcreteInline()
        plan = await skill.prepare("msg", _ctx())
        sim = await skill.dry_run(plan)
        assert sim.would_succeed is True
        assert sim.dependencies_available is True

    def test_skill_type_attribute(self) -> None:
        assert _ConcreteInline.skill_type == "inline"


class TestBackgroundSkill:
    async def test_can_handle_always_zero(self) -> None:
        skill = _ConcreteBackground()
        assert await skill.can_handle("anything", _ctx()) == 0.0
        assert await skill.can_handle("/kwork", _ctx()) == 0.0

    async def test_default_prepare(self) -> None:
        skill = _ConcreteBackground()
        plan = await skill.prepare("msg", _ctx())
        assert plan.skill_name == "test_bg"
        assert plan.skill_type == "background"

    async def test_default_dry_run(self) -> None:
        skill = _ConcreteBackground()
        plan = await skill.prepare("msg", _ctx())
        sim = await skill.dry_run(plan)
        assert sim.would_succeed is True

    def test_skill_type_attribute(self) -> None:
        assert _ConcreteBackground.skill_type == "background"

    def test_default_trigger_type(self) -> None:
        assert _ConcreteBackground.trigger_type == "interval"


class TestDelegatedSkill:
    async def test_can_handle(self) -> None:
        skill = _ConcreteDelegated()
        assert await skill.can_handle("/do thing", _ctx()) == 1.0
        assert await skill.can_handle("/other", _ctx()) == 0.0

    async def test_prepare_is_custom(self) -> None:
        skill = _ConcreteDelegated()
        plan = await skill.prepare("/do something", _ctx())
        assert "something" in plan.human_summary

    async def test_dry_run_succeeds_when_executor_configured(self) -> None:
        skill = _ConcreteDelegated()
        plan = await skill.prepare("/do thing", _ctx())
        sim = await skill.dry_run(plan)
        assert sim.would_succeed is True
        assert not sim.blockers

    async def test_dry_run_flags_missing_executor(self) -> None:
        class _NoExecutor(_ConcreteDelegated):
            executor: ClassVar[str] = ""

        skill = _NoExecutor()
        plan = await skill.prepare("/do thing", _ctx())
        sim = await skill.dry_run(plan)
        assert sim.would_succeed is False
        assert "executor not configured" in sim.blockers

    def test_skill_type_attribute(self) -> None:
        assert _ConcreteDelegated.skill_type == "delegated"


async def test_on_feedback_default_no_op() -> None:
    skill = _ConcreteInline()
    await skill.on_feedback(
        Feedback(skill_name="test_inline", user_id=1, rating="positive")
    )


def test_side_effect_enum_covers_major_surfaces() -> None:
    # Smoke check — protects against accidental enum value renames.
    assert SideEffect.CALLS_LLM.value == "calls_llm"
    assert SideEffect.POSTS_TO_CHANNEL.value == "posts_to_channel"
    assert SideEffect.DELEGATES_TO_CODE_AGENT.value == "delegates_to_code_agent"
