"""MVP gate for the Codex-only self-coding contract."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_mvp_gate_passes_current_repo() -> None:
    from scripts.check_self_coding_mvp import check_self_coding_mvp

    assert check_self_coding_mvp(Path.cwd()) == []


def test_mvp_gate_requires_spec_evidence_for_zhvusha_specs(tmp_path: Path) -> None:
    from scripts.check_self_coding_mvp import check_self_coding_mvp

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "2026-05-07-missing-evidence.yaml").write_text(
        "\n".join(
            [
                "slug: missing-evidence",
                "title: Missing evidence",
                "created_by: zhvusha",
                "tier: 1",
            ]
        ),
        encoding="utf-8",
    )

    errors = check_self_coding_mvp(tmp_path)

    assert any("source_provenance" in error for error in errors)
    assert any("rationale" in error for error in errors)
    assert any("preserve_behavior" in error for error in errors)
    assert any("allowed_simplifications" in error for error in errors)


def test_mvp_gate_requires_news_pillars_contract_files(tmp_path: Path) -> None:
    from scripts.check_self_coding_mvp import check_self_coding_mvp

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tmp_path / ".importlinter").write_text("", encoding="utf-8")
    tier3 = tmp_path / "scripts"
    tier3.mkdir()
    (tier3 / "check_tier3_protection.sh").write_text("", encoding="utf-8")

    errors = check_self_coding_mvp(tmp_path)

    assert any("src/news/models.py" in error for error in errors)
    assert any("src/pillars/models.py" in error for error in errors)
    assert any(".importlinter" in error for error in errors)
