"""Topic clustering and priority scoring tests for Phase 18."""

from __future__ import annotations

from datetime import UTC, datetime

from src.news import SourceItem, cluster_source_items
from src.pillars import PillarConfig


def test_mock_articles_form_cluster_representative_with_priority() -> None:
    items = [
        SourceItem(
            id=f"codex-{i}",
            source=f"source-{i % 3}",
            url=f"https://example.com/{i}",
            title=f"OpenAI Codex hooks update {i}",
            body="Codex hooks improve self-coding safety and architecture gates.",
            ts=datetime(2026, 5, 7, tzinfo=UTC),
            source_type="official_docs" if i == 0 else "blog",
            source_tier="A" if i == 0 else "B",
        )
        for i in range(50)
    ]
    pillars = PillarConfig.from_mapping(
        {
            "version": "test",
            "pillars": [
                {
                    "id": "self",
                    "name": "Самосовершенствование",
                    "weight": 1.0,
                    "description": "codex self-coding architecture",
                    "keywords": ["codex", "self-coding", "architecture"],
                }
            ],
        }
    )

    clusters = cluster_source_items(items, pillars=pillars)

    assert clusters
    assert clusters[0].title.startswith("OpenAI Codex hooks")
    assert clusters[0].final_priority > clusters[0].base_importance
    assert clusters[0].pillar_alignment["self"] == 1.0
