"""Codex read-only reviewer contract."""

from __future__ import annotations

from unittest.mock import AsyncMock


async def test_reviewer_runs_read_only_codex_for_tier3_specs() -> None:
    from src.skills.implement_spec.reviewer import CodexReadOnlyReviewer, ReviewRequest

    runner = AsyncMock(return_value="APPROVE: looks good")
    reviewer = CodexReadOnlyReviewer(runner=runner)

    verdict = await reviewer.review(
        ReviewRequest(
            slug="core-change",
            tier=3,
            spec_yaml="tier: 3",
            diff="diff --git a/src/skills/base.py b/src/skills/base.py",
            test_output="passed",
            audit_log={"backend": "codex_cli"},
        )
    )

    assert verdict.verdict == "approve"
    assert verdict.backend == "codex_cli"
    prompt = runner.await_args.args[0]
    assert "Tier 3" in prompt
    assert "architecture principles" in prompt


async def test_reviewer_uses_read_only_codex_runner_for_tier2() -> None:
    from src.skills.implement_spec.reviewer import CodexReadOnlyReviewer, ReviewRequest

    runner = AsyncMock(return_value="APPROVE: matches spec")
    reviewer = CodexReadOnlyReviewer(runner=runner)

    verdict = await reviewer.review(
        ReviewRequest(
            slug="feature",
            tier=2,
            spec_yaml="tier: 2",
            diff="diff",
            test_output="passed",
            audit_log={"backend": "codex_cli"},
        )
    )

    assert verdict.verdict == "approve"
    assert verdict.backend == "codex_cli"
    runner.assert_awaited_once()
