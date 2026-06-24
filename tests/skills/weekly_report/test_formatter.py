"""Weekly report formatter tests."""

from __future__ import annotations

from src.skills.weekly_report.formatter import (
    ReportTopic,
    WeeklyReportSnapshot,
    format_weekly_report,
)


def test_weekly_report_groups_topics_by_pillars() -> None:
    text = format_weekly_report(
        WeeklyReportSnapshot(
            topics=[
                ReportTopic(
                    cluster_key="codex",
                    title="Codex hooks",
                    summary="Self-coding update.",
                    final_priority=90,
                    pillar_alignment={"self_improvement": 0.9},
                ),
                ReportTopic(
                    cluster_key="clients",
                    title="AI clients",
                    summary="Client-facing topic.",
                    final_priority=70,
                    pillar_alignment={"money": 0.8},
                ),
            ],
            generated_drafts=2,
            archive_successes=1,
            archive_failures=1,
        )
    )

    assert "Самосовершенствование" in text
    assert "Codex hooks" in text
    assert "Деньги" in text
    assert "AI clients" in text
    assert "Черновики постов: 2" in text


def test_weekly_report_is_honest_when_empty() -> None:
    text = format_weekly_report(WeeklyReportSnapshot())

    assert "Темы в движении: 0" in text
    assert "движения не видно" in text
