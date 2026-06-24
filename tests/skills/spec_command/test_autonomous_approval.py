"""Autonomous self-coding approval contract tests."""

from __future__ import annotations

import pytest


def _spec_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "slug": "autonomous-low-risk-test",
        "title": "Autonomous low-risk test",
        "created_at": "2026-05-12T10:00:00+00:00",
        "created_by": "zhvusha",
        "tier": 1,
        "goal": "Добавить низкорисковую проверку автономного самокодинга.",
        "failing_test": {
            "file": "tests/skills/autonomous_self_coding/test_skill.py",
            "name": "test_autonomous_self_coding_can_start_safe_spec",
            "spec": "Safe Жвуша-created spec can be self-approved and started.",
        },
        "whitelist_paths": (
            "src/skills/autonomous_self_coding/skill.py",
            "tests/skills/autonomous_self_coding/test_skill.py",
        ),
        "blast_radius": ("new background self-coding orchestration only",),
        "rollback_path": ("git revert",),
        "preserve_behavior": (
            "Existing /код, spec, approval, whitelist, test and no-downgrade gates stay intact.",
        ),
        "allowed_simplifications": (),
        "rationale": "Generated from Жвуша self-work cycle evidence.",
        "source_provenance": (
            {
                "url": "workspace:self_coding_archive",
                "source_type": "local_repo",
                "trust_tier": "direct",
                "claim": "Autonomous self-work cycle selected a low-risk improvement.",
            },
        ),
    }
    base.update(overrides)
    return base


def test_spec_model_accepts_audited_zhvusha_approval() -> None:
    from src.skills.spec_command.parser import SpecModel, SpecStatus

    spec = SpecModel.model_validate(
        _spec_kwargs(
            status=SpecStatus.APPROVED,
            approved_at="2026-05-12T10:10:00+00:00",
            approved_by="zhvusha",
            autonomous_approval_reason=(
                "Tier 1, Жвуша-created, no live env activation, no Tier 3 paths."
            ),
        )
    )

    assert spec.approved_by == "zhvusha"
    assert spec.autonomous_approval_reason.startswith("Tier 1")


def test_spec_model_rejects_zhvusha_approval_without_reason() -> None:
    from pydantic import ValidationError
    from src.skills.spec_command.parser import SpecModel, SpecStatus

    with pytest.raises(ValidationError, match="autonomous_approval_reason"):
        SpecModel.model_validate(
            _spec_kwargs(
                status=SpecStatus.APPROVED,
                approved_at="2026-05-12T10:10:00+00:00",
                approved_by="zhvusha",
                autonomous_approval_reason="",
            )
        )


def test_spec_model_rejects_tier3_zhvusha_approval() -> None:
    from pydantic import ValidationError
    from src.skills.spec_command.parser import SpecModel, SpecStatus

    with pytest.raises(ValidationError, match="Tier 3 requires Никита approval"):
        SpecModel.model_validate(
            _spec_kwargs(
                status=SpecStatus.APPROVED,
                tier=3,
                whitelist_paths=("proposals/autonomous-core-change.md",),
                approved_at="2026-05-12T10:10:00+00:00",
                approved_by="zhvusha",
                autonomous_approval_reason=(
                    "Tier 3 autonomous mandate from Никита; post-implementation "
                    "architecture/safety review remains mandatory."
                ),
            )
        )
