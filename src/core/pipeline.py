"""PipelineRunner — generic primitive for linear pipelines inside a single module.

Used by v4 module-internal pipelines:

- incoming message pipeline (bot middleware)
- outgoing response pipeline (raw skill output → personality → safety → format)
- memory consolidation pipeline (raw episodes → enrichment → dedup → embed)
- LLM call pipeline (inside LLM Gateway)
- skill execution pipeline (permission → cost → approval → execute → log)

Not used for inter-module communication — that happens via service calls
through ``protocols.py``, not through a pipeline (see KB #69, section
"Pipeline pattern").

Design notes:

- Context type is a generic parameter (``ContextT``), not ``dict[str, Any]``.
  Typed contexts catch stage incompatibilities at review and mypy time, not at
  runtime (KB #69, "Защиты от ловушек pipeline").
- Stage validation runs once in ``PipelineRunner.__init__``, not on every
  ``run`` call — invalid pipelines fail fast with :class:`PipelineLoadError`.
- Stage duration is logged via structlog. When a stage returns the same
  context object it received, a warning is emitted (soft immutability hint).
- This module is a leaf — it must not import from other ``src/*`` modules
  (enforced manually; ``src.core`` is intentionally not covered by the phase-1
  layered-architecture import-linter rule).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog

_logger = structlog.get_logger(__name__)


class PreconditionError(Exception):
    """Raised when a stage's input context fails its preconditions check.

    Stage implementations raise this from ``validate_preconditions`` when the
    incoming context is not a valid input. The runner does not catch it — the
    error propagates to the caller of ``PipelineRunner.run``.
    """


class PostconditionError(Exception):
    """Raised when a stage's output context fails its postconditions check.

    Stage implementations raise this from ``validate_postconditions`` when the
    returned context is not a valid output. Signals a bug in the stage itself
    or corrupted upstream data.
    """


class PipelineLoadError(Exception):
    """Raised at ``PipelineRunner`` construction when the stage list is invalid.

    Fail-fast: all structural problems (non-stage instance, empty name,
    duplicate name) are reported before any ``run`` is attempted.
    """


class PipelineStage[ContextT](ABC):
    """Single stage in a typed pipeline.

    Subclasses MUST set a non-empty, unique ``name`` class attribute and
    implement :meth:`execute`. Optional hooks:

    * :meth:`validate_preconditions` — called with the input context before
      ``execute``; raise :class:`PreconditionError` on invalid input.
    * :meth:`validate_postconditions` — called with the returned context
      after ``execute``; raise :class:`PostconditionError` on invalid output.

    Stages should return a NEW context (``dataclasses.replace``,
    ``pydantic model_copy(update=...)``, etc.). Returning the same object is
    allowed but the runner will log a warning — it hints at mutation-in-place,
    which breaks the pipeline's immutability contract.
    """

    name: str = ""

    @abstractmethod
    async def execute(self, context: ContextT) -> ContextT:
        """Transform ``context`` and return the new context."""

    def validate_preconditions(self, context: ContextT) -> None:  # noqa: B027
        """Check input invariants. Default: no-op. Override to enforce."""

    def validate_postconditions(self, result: ContextT) -> None:  # noqa: B027
        """Check output invariants. Default: no-op. Override to enforce."""


class PipelineRunner[ContextT]:
    """Executes a fixed list of :class:`PipelineStage` instances sequentially.

    Validation runs once at construction time, not on each ``run`` call —
    invalid pipelines fail fast and loudly before any execution attempt.
    """

    def __init__(self, stages: list[PipelineStage[ContextT]]) -> None:
        self._stages: list[PipelineStage[ContextT]] = list(stages)
        self._validate_stages_on_load()

    def _validate_stages_on_load(self) -> None:
        seen_names: set[str] = set()
        for idx, stage in enumerate(self._stages):
            if not isinstance(stage, PipelineStage):
                raise PipelineLoadError(
                    f"stage at index {idx} is not a PipelineStage instance: "
                    f"got {type(stage).__name__}"
                )
            stage_name = getattr(stage, "name", "")
            if not stage_name:
                raise PipelineLoadError(
                    f"stage at index {idx} ({type(stage).__name__}) has empty name"
                )
            if stage_name in seen_names:
                raise PipelineLoadError(
                    f"duplicate stage name at index {idx}: {stage_name!r}"
                )
            seen_names.add(stage_name)

    async def run(self, initial_context: ContextT) -> ContextT:
        """Run all stages sequentially and return the final context.

        For each stage:

        1. ``validate_preconditions(context)`` is called.
        2. ``await execute(context)`` produces a new context.
        3. ``validate_postconditions(new_context)`` is called.
        4. Stage duration is logged via structlog.

        If a stage returns the same object it received, a WARNING is logged
        (soft immutability contract) but the pipeline continues.
        """
        context = initial_context
        for stage in self._stages:
            stage.validate_preconditions(context)
            started = time.monotonic()
            new_context = await stage.execute(context)
            duration_ms = (time.monotonic() - started) * 1000.0
            stage.validate_postconditions(new_context)
            if new_context is context:
                _logger.warning(
                    "pipeline.stage.returned_same_context",
                    stage=stage.name,
                    duration_ms=duration_ms,
                )
            else:
                _logger.debug(
                    "pipeline.stage.completed",
                    stage=stage.name,
                    duration_ms=duration_ms,
                )
            context = new_context
        return context


__all__ = [
    "PipelineLoadError",
    "PipelineRunner",
    "PipelineStage",
    "PostconditionError",
    "PreconditionError",
]
