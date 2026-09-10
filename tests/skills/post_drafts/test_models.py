"""Post draft model helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from src.skills.post_drafts.models import (
    PostTopic,
    build_post_draft,
    select_post_topics,
)


def test_select_post_topics_requires_money_alignment() -> None:
    selected = select_post_topics(
        [
            PostTopic(
                cluster_key="money",
                title="Kwork trend",
                summary="A client-facing AI trend.",
                final_priority=70,
                pillar_alignment={"money": 0.8},
            ),
            PostTopic(
                cluster_key="self",
                title="Codex hooks",
                summary="Internal self-coding update.",
                final_priority=95,
                pillar_alignment={"self_improvement": 1.0},
            ),
        ]
    )

    assert [topic.cluster_key for topic in selected] == ["money"]


def test_build_post_draft_keeps_source_cluster_and_text() -> None:
    draft = build_post_draft(
        PostTopic(
            cluster_key="ai-clients",
            title="AI clients",
            summary="New client-facing AI opportunity.",
            final_priority=80,
            pillar_alignment={"money": 0.9},
        ),
        now=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
    )

    assert draft.slug == "ai-clients"
    assert draft.source_cluster == "ai-clients"
    assert "New client-facing AI opportunity" in draft.text
