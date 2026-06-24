"""Adapters from existing collectors into ``SourceItem`` records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.news.models import SourceItem, make_source_item_id

if TYPE_CHECKING:
    from src.collectors.telegram_channels import ChannelMessage
    from src.collectors.youtube import YouTubeEntry


def telegram_messages_to_source_items(
    messages: list[ChannelMessage],
) -> list[SourceItem]:
    items: list[SourceItem] = []
    for message in messages:
        title = f"Telegram: {message.channel_title} #{message.message_id}"
        url = f"tg://channel/{message.channel_id}/{message.message_id}"
        body = message.text
        items.append(
            SourceItem(
                id=make_source_item_id(
                    "telegram-channels",
                    url,
                    title,
                    message.date,
                ),
                source="telegram-channels",
                url=url,
                title=title,
                body=body,
                ts=_ensure_utc(message.date),
                lang="ru",
                source_type="telegram",
                source_tier="D",
                metadata={
                    "channel_id": str(message.channel_id),
                    "channel_title": message.channel_title,
                    "message_id": str(message.message_id),
                    "views": str(message.views or ""),
                },
            )
        )
    return items


def youtube_entries_to_source_items(entries: list[YouTubeEntry]) -> list[SourceItem]:
    items: list[SourceItem] = []
    for entry in entries:
        ts = entry.watched_at or datetime.now(tz=UTC)
        title = f"YouTube: {entry.title}"
        body = f"Channel: {entry.channel or 'unknown'}"
        if entry.duration:
            body += f"\nDuration: {entry.duration}"
        items.append(
            SourceItem(
                id=make_source_item_id("youtube", entry.url, title, ts),
                source="youtube",
                url=entry.url,
                title=title,
                body=body,
                ts=_ensure_utc(ts),
                lang="en",
                source_type="youtube",
                source_tier="D",
                metadata={
                    "video_id": entry.video_id,
                    "channel": entry.channel,
                    "duration": entry.duration or "",
                },
            )
        )
    return items


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
