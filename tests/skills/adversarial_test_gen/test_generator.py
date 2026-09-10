"""Anchored adversarial test generation from archive failures."""

from __future__ import annotations

from datetime import UTC, datetime

from src.archive.models import ArchiveNode, ArchiveStatus


def _node(
    slug: str, insight: str, *, status: ArchiveStatus = ArchiveStatus.FAILED
) -> ArchiveNode:
    return ArchiveNode(
        slug=slug,
        spec_slug=slug.removesuffix("-node"),
        tier=2,
        status=status,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
        diff_summary="Cycle stopped before commit.",
        tests_summary="pytest failed",
        insight=insight,
        tags=["self-coding", "failed", "contract-drift"],
    )


def test_adversarial_tests_are_anchored_to_archive_nodes() -> None:
    from src.skills.adversarial_test_gen.generator import generate_adversarial_tests

    drafts = generate_adversarial_tests(
        [
            _node("a-node", "HTML response was double-converted in Telegram."),
            _node("b-node", "HTML response rendered as literal tags."),
            _node("c-node", "HTML formatter drift broke chat output."),
        ],
        limit=3,
    )

    assert len(drafts) == 3
    for draft in drafts:
        assert draft.archive_slug in {"a-node", "b-node", "c-node"}
        assert draft.test_name.startswith("test_adversarial_")
        assert "archive node" in draft.body.lower()
        assert draft.archive_slug in draft.body


def test_successful_archive_nodes_are_not_used_by_default() -> None:
    from src.skills.adversarial_test_gen.generator import generate_adversarial_tests

    drafts = generate_adversarial_tests(
        [
            _node("good-node", "successful path", status=ArchiveStatus.COMMITTED),
            _node("bad-node", "failed path"),
        ]
    )

    assert [draft.archive_slug for draft in drafts] == ["bad-node"]
