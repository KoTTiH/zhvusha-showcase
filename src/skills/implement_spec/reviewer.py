"""Codex read-only reviewer for completed self-coding artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ReviewVerdictValue = Literal["approve", "reject", "escalate"]


@dataclass(frozen=True)
class ReviewRequest:
    slug: str
    tier: int
    spec_yaml: str
    diff: str
    test_output: str
    audit_log: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewVerdict:
    verdict: ReviewVerdictValue
    rationale: str
    backend: str = "codex_cli"


class ReviewRunner(Protocol):
    async def __call__(self, prompt: str) -> str: ...


class CodexReadOnlyReviewer:
    """Review artifacts through a read-only Codex runner.

    Tier 3 is reviewed by the same read-only runner, but the prompt marks it as
    a high-risk architecture change and demands explicit blocker reporting.
    """

    def __init__(self, *, runner: ReviewRunner) -> None:
        self._runner = runner

    async def review(self, request: ReviewRequest) -> ReviewVerdict:
        raw = await self._runner(_build_prompt(request))
        lowered = raw.strip().lower()
        if lowered.startswith("approve"):
            return ReviewVerdict(verdict="approve", rationale=raw.strip())
        if lowered.startswith("reject"):
            return ReviewVerdict(verdict="reject", rationale=raw.strip())
        return ReviewVerdict(verdict="escalate", rationale=raw.strip() or "No verdict.")


def _build_prompt(request: ReviewRequest) -> str:
    return "\n\n".join(
        [
            "You are a read-only Codex reviewer. Do not edit files.",
            (
                "For Tier 3, be stricter: check architecture principles, "
                "safety, secrets/env guard, side effects, no-downgrade, "
                "rollback path and ban/API risk. Return REJECT or ESCALATE "
                "if any blocker remains."
            ),
            f"Slug: {request.slug}",
            f"Tier: {request.tier}",
            "# Spec",
            request.spec_yaml,
            "# Diff",
            request.diff,
            "# Tests",
            request.test_output,
            "# Audit",
            str(request.audit_log),
            "Return APPROVE, REJECT, or ESCALATE with a short rationale.",
        ]
    )
