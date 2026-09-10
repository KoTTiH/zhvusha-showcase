"""Contract tests for src.core.pipeline.PipelineRunner (KB #69, #70).

These tests exercise the PipelineRunner primitive in isolation. Runner is a
generic async executor over PipelineStage instances; it's the foundation for
all v4 module pipelines (incoming message, outgoing response, memory
consolidation, LLM call, skill execution).

Marker: contract — level 1 of the v4 test pyramid.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest
from src.core.pipeline import (
    PipelineLoadError,
    PipelineRunner,
    PipelineStage,
    PostconditionError,
    PreconditionError,
)

pytestmark = pytest.mark.contract


@dataclass(frozen=True)
class Ctx:
    """Minimal immutable context used by the test stages."""

    value: int


# --- Test stages -----------------------------------------------------------


class AddOne(PipelineStage[Ctx]):
    name = "add_one"

    async def execute(self, context: Ctx) -> Ctx:
        return replace(context, value=context.value + 1)


class MulTwo(PipelineStage[Ctx]):
    name = "mul_two"

    async def execute(self, context: Ctx) -> Ctx:
        return replace(context, value=context.value * 2)


class AddTen(PipelineStage[Ctx]):
    name = "add_ten"

    async def execute(self, context: Ctx) -> Ctx:
        return replace(context, value=context.value + 10)


class RefusePositive(PipelineStage[Ctx]):
    name = "refuse_positive"

    async def execute(self, context: Ctx) -> Ctx:
        return context

    def validate_preconditions(self, context: Ctx) -> None:
        if context.value > 0:
            raise PreconditionError(f"value must be <= 0, got {context.value}")


class BadOutput(PipelineStage[Ctx]):
    name = "bad_output"

    async def execute(self, context: Ctx) -> Ctx:
        return replace(context, value=-1)

    def validate_postconditions(self, result: Ctx) -> None:
        if result.value < 0:
            raise PostconditionError(f"value must be non-negative, got {result.value}")


class AsyncDelayStage(PipelineStage[Ctx]):
    name = "async_delay"

    async def execute(self, context: Ctx) -> Ctx:
        await asyncio.sleep(0.01)
        return replace(context, value=context.value + 100)


class IdentityStage(PipelineStage[Ctx]):
    """Returns the exact same context object — used for immutability hint test."""

    name = "identity"

    async def execute(self, context: Ctx) -> Ctx:
        return context


# --- Tests -----------------------------------------------------------------


async def test_happy_path_three_stages_compose_in_order() -> None:
    """3 stages apply sequentially; (0+1)*2+10 == 12."""
    runner: PipelineRunner[Ctx] = PipelineRunner([AddOne(), MulTwo(), AddTen()])
    result = await runner.run(Ctx(value=0))
    assert result.value == 12


async def test_precondition_violation_is_raised() -> None:
    runner: PipelineRunner[Ctx] = PipelineRunner([RefusePositive()])
    with pytest.raises(PreconditionError, match="value must be <= 0"):
        await runner.run(Ctx(value=5))


async def test_postcondition_violation_is_raised() -> None:
    runner: PipelineRunner[Ctx] = PipelineRunner([BadOutput()])
    with pytest.raises(PostconditionError, match="non-negative"):
        await runner.run(Ctx(value=0))


async def test_empty_pipeline_returns_context_unchanged() -> None:
    runner: PipelineRunner[Ctx] = PipelineRunner([])
    ctx = Ctx(value=42)
    result = await runner.run(ctx)
    assert result.value == 42
    # empty pipeline returns the exact same object
    assert result is ctx


async def test_single_stage_pipeline_runs() -> None:
    runner: PipelineRunner[Ctx] = PipelineRunner([AddOne()])
    result = await runner.run(Ctx(value=10))
    assert result.value == 11


async def test_async_stage_with_real_delay_completes() -> None:
    runner: PipelineRunner[Ctx] = PipelineRunner([AsyncDelayStage()])
    result = await runner.run(Ctx(value=0))
    assert result.value == 100


def test_duplicate_stage_names_raise_load_error_at_construction() -> None:
    """Load-time validation: two stages with the same name must fail fast."""
    with pytest.raises(PipelineLoadError, match="duplicate"):
        PipelineRunner([AddOne(), AddOne()])


def test_non_stage_instance_raises_load_error() -> None:
    """Only PipelineStage subclasses are accepted."""

    class NotAStage:
        name = "not_a_stage"

    with pytest.raises(PipelineLoadError):
        PipelineRunner([NotAStage()])  # type: ignore[arg-type]


def test_stage_with_empty_name_raises_load_error() -> None:
    class NoNameStage(PipelineStage[Ctx]):
        name = ""

        async def execute(self, context: Ctx) -> Ctx:
            return context

    with pytest.raises(PipelineLoadError, match="name"):
        PipelineRunner([NoNameStage()])


async def test_identity_stage_does_not_raise() -> None:
    """A stage returning the same object is allowed (mutability is a soft contract).

    The runner may log a warning but must not fail — pipeline should still complete.
    """
    runner: PipelineRunner[Ctx] = PipelineRunner([IdentityStage()])
    ctx = Ctx(value=1)
    result = await runner.run(ctx)
    assert result.value == 1
