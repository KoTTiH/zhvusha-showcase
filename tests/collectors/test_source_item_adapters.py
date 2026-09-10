"""Existing collectors adapted into SourceItem records for Phase 20."""

from __future__ import annotations

from datetime import UTC, datetime

from src.collectors.source_items import (
    telegram_messages_to_source_items,
    youtube_entries_to_source_items,
)
from src.collectors.telegram_channels import ChannelMessage
from src.collectors.youtube import YouTubeEntry


def test_telegram_channel_message_becomes_source_item() -> None:
    items = telegram_messages_to_source_items(
        [
            ChannelMessage(
                channel_id=10,
                channel_title="TechSparks",
                message_id=5,
                text="OpenAI обновил Codex hooks.",
                date=datetime(2026, 5, 7, 7, tzinfo=UTC),
                views=100,
            )
        ]
    )

    assert items[0].source == "telegram-channels"
    assert items[0].source_type == "telegram"
    assert items[0].source_tier == "D"
    assert items[0].metadata["channel_title"] == "TechSparks"


def test_youtube_entry_becomes_source_item() -> None:
    items = youtube_entries_to_source_items(
        [
            YouTubeEntry(
                video_id="abc",
                title="Agent safety talk",
                channel="Sakana",
                url="https://youtube.com/watch?v=abc",
                watched_at=datetime(2026, 5, 7, 7, tzinfo=UTC),
            )
        ]
    )

    assert items[0].source == "youtube"
    assert items[0].source_type == "youtube"
    assert items[0].source_tier == "D"
    assert items[0].metadata["video_id"] == "abc"
