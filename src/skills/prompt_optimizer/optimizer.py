"""Safe prompt optimization proposals from archive metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptMetric:
    template_id: str
    runs: int
    successes: int
    avg_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs > 0 else 0.0


@dataclass(frozen=True)
class PromptOptimizationProposal:
    current_template_id: str
    candidate_template_id: str
    current_success_rate: float
    candidate_success_rate: float
    requires_approval: bool = True


def propose_prompt_update(
    metrics: list[PromptMetric],
    *,
    current_template_id: str,
    min_runs: int = 3,
    min_delta: float = 0.15,
) -> PromptOptimizationProposal | None:
    """Propose, but never apply, a better prompt template."""
    eligible = [metric for metric in metrics if metric.runs >= min_runs]
    current = next(
        (metric for metric in eligible if metric.template_id == current_template_id),
        None,
    )
    if current is None:
        return None
    candidate = max(eligible, key=lambda metric: metric.success_rate, default=current)
    if candidate.template_id == current.template_id:
        return None
    if candidate.success_rate - current.success_rate < min_delta:
        return None
    return PromptOptimizationProposal(
        current_template_id=current.template_id,
        candidate_template_id=candidate.template_id,
        current_success_rate=current.success_rate,
        candidate_success_rate=candidate.success_rate,
    )
