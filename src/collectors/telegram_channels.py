"""Telegram channel collector via Telethon."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from types import SimpleNamespace

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()


@dataclass
class ChannelMessage:
    channel_id: int
    channel_title: str
    message_id: int
    text: str
    date: datetime
    views: int | None


class TelegramChannelCollector:
    """Monitors Telegram channels via Telethon userbot.

    Connects only during collection, disconnects immediately after.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        self._api_id = getattr(config, "telegram_api_id", 0)
        self._api_hash = getattr(config, "telegram_api_hash", "")
        self._session_path = getattr(
            config, "telethon_session_path", "~/.zhvusha_telethon.session"
        )
        self._workspace = Path(
            getattr(config, "workspace_path", "~/zhvusha-workspace")
        ).expanduser()
        self._admin_user_id = getattr(config, "admin_user_id", 0)
        self._delay = getattr(config, "channel_read_delay_seconds", 1.5)

        raw_ids = getattr(config, "monitored_channel_ids", "")
        self._channel_ids: list[str] = [
            c.strip() for c in raw_ids.split(",") if c.strip()
        ]

        self._client: Any = None
        self._connected = False

    async def connect(self) -> None:
        """Connect Telethon client."""
        try:
            self._client = self._create_client()
            await self._client.connect()
            self._connected = True
            logger.info("telegram_channels_connected")
        except Exception:
            logger.warning("telegram_channels_connect_failed", exc_info=True)
            self._connected = False

    def _create_client(self) -> Any:
        """Create Telethon TelegramClient. Separated for testability."""
        from telethon import TelegramClient

        session_path = Path(self._session_path).expanduser()
        return TelegramClient(str(session_path), self._api_id, self._api_hash)

    async def disconnect(self) -> None:
        """Disconnect Telethon client."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                logger.warning("telegram_channels_disconnect_failed", exc_info=True)
            self._connected = False

    async def collect_messages(
        self,
        since: datetime | None = None,
        limit_per_channel: int | None = None,
    ) -> dict[str, list[ChannelMessage]]:
        """Collect messages from all monitored channels.

        By default the caller's `since` window is the only bound. `/morning`
        recovery runs can span many days, so a fixed per-channel cap would
        silently drop older messages from active chats.
        """
        if since is None:
            since = datetime.now(tz=UTC) - timedelta(hours=24)

        result: dict[str, list[ChannelMessage]] = {}

        for channel_id in self._channel_ids:
            try:
                messages = await self._fetch_channel_messages(
                    channel_id, since=since, limit=limit_per_channel
                )
                if messages:
                    result[channel_id] = messages

                if self._delay > 0:
                    await asyncio.sleep(self._delay)

            except Exception:
                logger.warning(
                    "telegram_channel_fetch_failed",
                    channel=channel_id,
                    exc_info=True,
                )
                continue

        logger.info(
            "telegram_channels_collected",
            channels=len(result),
            total_messages=sum(len(m) for m in result.values()),
        )
        return result

    async def _fetch_channel_messages(
        self,
        channel_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[ChannelMessage]:
        """Fetch messages from a single channel via Telethon."""
        if not self._connected or self._client is None:
            return []

        try:
            parsed_id: int | str = int(channel_id)
        except ValueError:
            parsed_id = channel_id
        entity = await self._client.get_entity(parsed_id)
        messages: list[ChannelMessage] = []

        async for msg in self._client.iter_messages(entity, limit=limit):
            if since and msg.date < since:
                break
            if not msg.text:
                continue

            messages.append(
                ChannelMessage(
                    channel_id=getattr(entity, "id", 0),
                    channel_title=getattr(entity, "title", channel_id),
                    message_id=msg.id,
                    text=msg.text,
                    date=msg.date,
                    views=getattr(msg, "views", None),
                )
            )

        return messages

    def _format_messages(
        self,
        messages: dict[str, list[ChannelMessage]],
    ) -> str:
        """Format raw messages as markdown for the morning session to analyze."""
        sections: list[str] = []

        for channel_id, channel_msgs in messages.items():
            if not channel_msgs:
                continue

            title = channel_msgs[0].channel_title if channel_msgs else channel_id
            lines = [f"## {title} ({len(channel_msgs)} сообщ.)", ""]

            for msg in channel_msgs:
                time_str = msg.date.strftime("%H:%M")
                lines.append(f"**[{time_str}]** {msg.text}")
                if msg.views is not None:
                    lines.append(f"👁 {msg.views}")
                lines.append("")

            sections.append("\n".join(lines))

        return "\n".join(sections) if sections else "Нет сообщений в каналах."

    async def collect_and_save(
        self,
        episodic: EpisodicMemory | None = None,
        since: datetime | None = None,
    ) -> str:
        """Collect messages and write raw text to the workspace inbox."""
        messages = await self.collect_messages(since=since)

        if not messages:
            return "Нет сообщений в каналах."

        formatted = self._format_messages(messages)
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        total = sum(len(msgs) for msgs in messages.values())

        # Write raw messages to inbox; analysis is done by the morning session.
        inbox_dir = self._workspace / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        inbox_path = inbox_dir / f"channels_{today}.md"
        inbox_path.write_text(
            f"# Telegram Channels — {today}\n\n{formatted}",
            encoding="utf-8",
        )
        logger.info("telegram_inbox_written", path=str(inbox_path))

        # Record episode about channel collection
        if episodic is not None:
            channel_names = [
                msgs[0].channel_title for msgs in messages.values() if msgs
            ]
            await episodic.record(
                content=(
                    f"Собрано {total} сообщений из каналов: {', '.join(channel_names)}"
                ),
                user_id=self._admin_user_id,
                chat_type="personal",
                role="assistant",
                source="channel",
                importance=0.4,
                person_name="Жвуша",
                significance="inner_circle",
                domain="content",
            )

        return f"{total} сообщений из {len(messages)} каналов"
