"""CycleAnalyzerSkill contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

from src.archive.models import ArchiveNode, ArchiveStatus
from src.skills.base import AgentContext, InlineSkill
from src.skills.cycle_analyzer.skill import CycleAnalyzerSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)


class _Store:
    async def lookup(self, query: str, *, top_k: int = 5) -> list[ArchiveNode]:
        del query, top_k
        return [
            ArchiveNode(
                slug="codex-hooks-abc123",
                spec_slug="codex-hooks",
                tier=2,
                status=ArchiveStatus.COMMITTED,
                created_at=datetime(2026, 5, 7, tzinfo=UTC),
                commit_sha="abc1234567890",
                diff_summary="Codex hooks",
                tests_summary="tests passed",
                insight="Codex hooks path worked.",
            )
        ]


def _ctx() -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal")


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(CycleAnalyzerSkill)
    validate_manifest_matches_class(manifest, CycleAnalyzerSkill)
    assert issubclass(CycleAnalyzerSkill, InlineSkill)


async def test_lookup_returns_archive_hits() -> None:
    skill = CycleAnalyzerSkill(admin_user_id=1, archive_store=_Store())  # type: ignore[arg-type]

    result = await skill.execute("/archive_lookup codex", _ctx())

    assert result.success
    assert "codex-hooks-abc123" in result.response
    assert result.metadata["count"] == 1


async def test_missing_store_returns_error() -> None:
    skill = CycleAnalyzerSkill(admin_user_id=1, archive_store=None)

    result = await skill.execute("/archive_lookup codex", _ctx())

    assert not result.success
    assert "Archive store" in result.response
