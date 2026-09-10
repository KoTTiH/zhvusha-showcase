"""Skill contract for adversarial_test_gen."""

from __future__ import annotations

from datetime import UTC, datetime

from src.archive.models import ArchiveNode, ArchiveStatus
from src.skills.base import AgentContext


class _Archive:
    async def lookup(self, query: str, *, top_k: int = 5) -> list[ArchiveNode]:
        del query, top_k
        return [
            ArchiveNode(
                slug="failed-html",
                spec_slug="html-fix",
                tier=2,
                status=ArchiveStatus.FAILED,
                created_at=datetime(2026, 5, 7, tzinfo=UTC),
                diff_summary="failed",
                tests_summary="pytest failed",
                insight="HTML output drift",
                tags=["failed"],
            )
        ]


def _ctx() -> AgentContext:
    return AgentContext(user_id=12345, chat_id=12345, mode="personal")


async def test_skill_renders_anchored_drafts() -> None:
    from src.skills.adversarial_test_gen.skill import AdversarialTestGenSkill

    skill = AdversarialTestGenSkill(admin_user_id=12345, archive_store=_Archive())  # type: ignore[arg-type]
    result = await skill.execute("/adversarial_tests html", _ctx())

    assert result.success
    assert "failed-html" in result.response
    assert result.metadata["count"] == 1


async def test_skill_requires_archive_store() -> None:
    from src.skills.adversarial_test_gen.skill import AdversarialTestGenSkill

    skill = AdversarialTestGenSkill(admin_user_id=12345, archive_store=None)
    result = await skill.execute("/adversarial_tests html", _ctx())

    assert not result.success
    assert "Archive store" in result.response
