"""Formal non-LLM gates for post-cycle self-coding review."""

from __future__ import annotations


def test_pre_commit_blocks_when_test_count_decreases() -> None:
    from src.skills.implement_spec.formal_gates import (
        FormalGateInputs,
        run_formal_gates,
    )

    verdict = run_formal_gates(
        FormalGateInputs(
            baseline_test_count=120,
            current_test_count=119,
            changed_files=["src/skills/weather/skill.py"],
            whitelist_paths=["src/skills/weather/skill.py"],
            existing_tests_to_update_paths=[],
            audit_log={"slug": "weather", "backend": "codex_cli"},
        )
    )

    assert not verdict.passed
    assert any("test count" in issue.lower() for issue in verdict.issues)


def test_formal_gates_reject_claude_backend_metadata() -> None:
    from src.skills.implement_spec.formal_gates import (
        FormalGateInputs,
        run_formal_gates,
    )

    verdict = run_formal_gates(
        FormalGateInputs(
            baseline_test_count=1,
            current_test_count=1,
            changed_files=["src/x.py"],
            whitelist_paths=["src/x.py"],
            existing_tests_to_update_paths=[],
            audit_log={"slug": "x", "backend": "claude_cli"},
        )
    )

    assert not verdict.passed
    assert any("claude" in issue.lower() for issue in verdict.issues)
