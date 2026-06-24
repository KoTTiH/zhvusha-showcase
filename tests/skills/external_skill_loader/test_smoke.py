"""External skill smoke/e2e checks."""

from __future__ import annotations

from pathlib import Path


async def test_external_skill_smoke_checker_runs_isolated_flow(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.smoke import ExternalSkillSmokeChecker

    scratch_root = tmp_path / "smoke"
    report = await ExternalSkillSmokeChecker(scratch_root=scratch_root).run_isolated()

    assert report.success is True
    assert {
        "quarantine_import",
        "registry_review",
        "readonly_runtime",
        "execution_runtime",
        "native_conversion",
        "doctor_status",
    }.issubset({step.code for step in report.steps})
    assert (scratch_root / "latest" / "registry").exists()
    assert not (tmp_path / "skills" / "external" / "registry").exists()

    rendered = report.render_for_operator()
    assert "External Skill Smoke status" in rendered
    assert "PASS" in rendered
    assert "approval-" not in rendered
