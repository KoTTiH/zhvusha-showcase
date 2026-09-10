"""Prompt optimization from archive metrics."""

from __future__ import annotations


def test_prompt_optimizer_proposes_candidate_from_archive_metrics() -> None:
    from src.skills.prompt_optimizer.optimizer import (
        PromptMetric,
        propose_prompt_update,
    )

    proposal = propose_prompt_update(
        [
            PromptMetric(
                template_id="architect.v1", runs=10, successes=4, avg_cost=0.2
            ),
            PromptMetric(
                template_id="architect.v2", runs=8, successes=7, avg_cost=0.25
            ),
        ],
        current_template_id="architect.v1",
    )

    assert proposal is not None
    assert proposal.current_template_id == "architect.v1"
    assert proposal.candidate_template_id == "architect.v2"
    assert proposal.requires_approval is True
