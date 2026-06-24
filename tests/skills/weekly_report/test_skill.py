"""WeeklyReportSkill contract and behavior tests."""

from __future__ import annotations

from src.skills.base import AgentContext, InlineSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.weekly_report.formatter import ReportTopic, WeeklyReportSnapshot
from src.skills.weekly_report.skill import WeeklyReportSkill


class _Provider:
    async def build_snapshot(self, *, days: int = 7) -> WeeklyReportSnapshot:
        return WeeklyReportSnapshot(
            days=days,
            topics=[
                ReportTopic(
                    cluster_key="codex",
                    title="Codex hooks",
                    summary="Self-coding update.",
                    final_priority=90,
                    pillar_alignment={"self_improvement": 0.9},
                )
            ],
        )


def _ctx() -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal")


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(WeeklyReportSkill)
    validate_manifest_matches_class(manifest, WeeklyReportSkill)
    assert issubclass(WeeklyReportSkill, InlineSkill)


async def test_execute_returns_weekly_report() -> None:
    skill = WeeklyReportSkill(admin_user_id=1, report_provider=_Provider())

    result = await skill.execute("/weekly_report 14", _ctx())

    assert result.success
    assert "Codex hooks" in result.response
    assert result.metadata["days"] == 14
