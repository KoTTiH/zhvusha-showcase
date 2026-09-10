"""Tier A source collectors for Phase 19."""

from __future__ import annotations

from datetime import UTC, datetime

from src.collectors.github_trending import parse_ossinsight_repos, repos_to_source_items
from src.collectors.huggingface import (
    models_to_source_items,
    parse_huggingface_models,
)
from src.collectors.lmarena import arena_models_to_source_items, parse_lmarena_models


def test_github_trending_payload_normalizes_to_source_item() -> None:
    repos = parse_ossinsight_repos(
        {
            "data": [
                {"full_name": "openai/codex", "description": "CLI agent", "stars": 42}
            ]
        }
    )

    items = repos_to_source_items(repos, ts=datetime(2026, 5, 7, tzinfo=UTC))

    assert len(items) == 1
    assert items[0].source == "github-trending"
    assert items[0].source_type == "github"
    assert items[0].source_tier == "A"
    assert items[0].url == "https://github.com/openai/codex"


def test_huggingface_payload_normalizes_to_source_item() -> None:
    models = parse_huggingface_models(
        [
            {
                "modelId": "openai/gpt-test",
                "downloads": 1000,
                "likes": 5,
                "lastModified": "2026-05-07T00:00:00Z",
                "tags": ["text-generation"],
            }
        ]
    )

    items = models_to_source_items(models)

    assert items[0].source == "huggingface-models"
    assert items[0].source_tier == "A"
    assert items[0].metadata["downloads"] == "1000"


def test_lmarena_payload_normalizes_to_source_item() -> None:
    models = parse_lmarena_models(
        {"leaderboard": [{"model": "gpt-5.5", "rank": 1, "arena_score": 1400}]}
    )

    items = arena_models_to_source_items(
        models,
        source_url="https://example.com/arena.json",
        ts=datetime(2026, 5, 7, tzinfo=UTC),
    )

    assert items[0].source == "lm-arena"
    assert items[0].title == "LM Arena: gpt-5.5 #1"
    assert items[0].source_tier == "A"
