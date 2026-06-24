"""Dedup pipeline tests for Phase 17."""

from __future__ import annotations

from datetime import UTC, datetime

from src.news.dedup import deduplicate_source_items
from src.news.models import SourceItem


def _item(id_: str, title: str, url: str, body: str) -> SourceItem:
    return SourceItem(
        id=id_,
        source="test",
        url=url,
        title=title,
        body=body,
        ts=datetime(2026, 5, 7, tzinfo=UTC),
    )


def test_dedup_removes_tracking_url_duplicates() -> None:
    result = deduplicate_source_items(
        [
            _item(
                "a", "OpenAI Codex hooks", "https://example.com/post?utm_source=x", "A"
            ),
            _item("b", "OpenAI Codex hooks", "https://example.com/post", "B"),
        ]
    )

    assert [item.id for item in result.unique_items] == ["a"]
    assert result.duplicates[0].duplicate_of == "a"
    assert result.duplicates[0].reason == "url"


def test_dedup_collapses_cross_lingual_ai_event_with_shared_entities() -> None:
    result = deduplicate_source_items(
        [
            _item(
                "en",
                "OpenAI Codex hooks improve coding agent safety",
                "https://example.com/en",
                "Codex hooks add deterministic lifecycle checks for agents.",
            ),
            _item(
                "ru",
                "OpenAI Codex хуки улучшают безопасность coding agents",
                "https://example.com/ru",
                "Codex hooks добавляют deterministic checks для агентов.",
            ),
        ],
        same_topic_threshold=0.35,
    )

    assert [item.id for item in result.unique_items] == ["en"]
    assert result.duplicates[0].duplicate_of == "en"
