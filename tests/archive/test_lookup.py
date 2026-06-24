"""Archive lookup ranking tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.archive.models import ArchiveNode, ArchiveStatus
from src.archive.store import archive_lookup


def _node(slug: str, text: str, offset: int) -> ArchiveNode:
    return ArchiveNode(
        slug=slug,
        spec_slug=slug,
        tier=1,
        status=ArchiveStatus.COMMITTED,
        created_at=datetime(2026, 5, 7, tzinfo=UTC) + timedelta(minutes=offset),
        diff_summary=text,
        tests_summary="tests passed",
        insight=text,
    )


def test_archive_lookup_ranks_by_token_overlap_then_recency() -> None:
    older = _node("codex-hooks", "codex hooks self coding", 0)
    newer = _node("codex-cli", "codex cli backend", 1)
    unrelated = _node("weather", "weather skill", 2)

    result = archive_lookup("codex hooks", [unrelated, newer, older], top_k=2)

    assert [node.slug for node in result] == ["codex-hooks", "codex-cli"]


def test_archive_lookup_searches_metadata_context() -> None:
    greeting = _node("greeting-calibration", "prompt calibration", 0)
    greeting.metadata = {
        "chat_context": ["Никита: Жвуша переигрывает живость на бытовых приветствиях"]
    }
    unrelated = _node("weather", "weather skill", 1)

    result = archive_lookup("переигрывает приветствия", [unrelated, greeting], top_k=1)

    assert [node.slug for node in result] == ["greeting-calibration"]
