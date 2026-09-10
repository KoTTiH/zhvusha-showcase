"""Tests for Telegram channel collector."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.collectors.telegram_channels import (
    ChannelMessage,
    TelegramChannelCollector,
)


@pytest.fixture
def tg_settings(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    (workspace / "inbox").mkdir(parents=True)
    (workspace / "knowledge" / "channels").mkdir(parents=True)
    return SimpleNamespace(
        workspace_path=str(workspace),
        telegram_api_id=12345,
        telegram_api_hash="fake_hash",
        telethon_session_path=str(tmp_path / "test.session"),
        monitored_channel_ids="@test_channel,@another_channel",
        channel_read_delay_seconds=0.0,  # No delay in tests
        admin_user_id=12345,
    )


@pytest.fixture
def mock_telethon_client() -> AsyncMock:
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.is_connected = MagicMock(return_value=True)
    client.get_entity = AsyncMock()
    client.iter_messages = MagicMock()
    return client


def _make_messages(channel_title: str, texts: list[str]) -> list[ChannelMessage]:
    """Create test channel messages."""
    return [
        ChannelMessage(
            channel_id=-100123,
            channel_title=channel_title,
            message_id=i,
            text=text,
            date=datetime(2026, 4, 1, 10 + i, 0, tzinfo=UTC),
            views=100 * i,
        )
        for i, text in enumerate(texts, 1)
    ]


class _FakeTelegramMessage:
    def __init__(self, *, message_id: int, text: str, date: datetime) -> None:
        self.id = message_id
        self.text = text
        self.date = date
        self.views = None


class _AsyncMessageIter:
    def __init__(self, messages: list[_FakeTelegramMessage]) -> None:
        self._messages = messages
        self._index = 0
        self.consumed = 0

    def __aiter__(self) -> _AsyncMessageIter:
        return self

    async def __anext__(self) -> _FakeTelegramMessage:
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        item = self._messages[self._index]
        self._index += 1
        self.consumed += 1
        return item


async def test_connect_and_disconnect(
    tg_settings: SimpleNamespace, mock_telethon_client: AsyncMock
):
    """Telethon client connects and disconnects properly."""
    collector = TelegramChannelCollector(tg_settings)

    with patch.object(collector, "_create_client", return_value=mock_telethon_client):
        await collector.connect()
        await collector.disconnect()

    mock_telethon_client.connect.assert_awaited_once()
    mock_telethon_client.disconnect.assert_awaited_once()


async def test_collect_messages_from_channels(
    tg_settings: SimpleNamespace,
):
    """Messages are collected from all monitored channels."""
    collector = TelegramChannelCollector(tg_settings)

    messages = {
        "@test_channel": _make_messages(
            "Test Channel",
            [
                "A" * 60,  # Long enough (>50 chars)
                "B" * 60,
            ],
        ),
        "@another_channel": _make_messages(
            "Another",
            [
                "C" * 60,
            ],
        ),
    }

    with patch.object(
        collector, "_fetch_channel_messages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = lambda ch, **kw: messages.get(ch, [])
        result = await collector.collect_messages()

    assert len(result) == 2
    assert len(result["@test_channel"]) == 2
    assert len(result["@another_channel"]) == 1

    for call in mock_fetch.call_args_list:
        assert call.kwargs["limit"] is None


async def test_fetch_channel_messages_reads_full_since_window_without_default_cap(
    tg_settings: SimpleNamespace,
):
    """Default fetch reads all messages until `since`, not only first 50."""
    collector = TelegramChannelCollector(tg_settings)
    collector._connected = True

    client = AsyncMock()
    client.get_entity = AsyncMock(
        return_value=SimpleNamespace(id=-100123, title="Big Chat")
    )
    since = datetime(2026, 5, 20, 10, 0, tzinfo=UTC)
    valid_messages = [
        _FakeTelegramMessage(
            message_id=i,
            text=f"message {i}",
            date=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        )
        for i in range(120)
    ]
    older_empty = _FakeTelegramMessage(
        message_id=1000,
        text="",
        date=datetime(2026, 5, 20, 9, 59, tzinfo=UTC),
    )
    older_text = _FakeTelegramMessage(
        message_id=1001,
        text="too old",
        date=datetime(2026, 5, 20, 9, 58, tzinfo=UTC),
    )

    iterator = _AsyncMessageIter([*valid_messages, older_empty, older_text])

    def iter_messages(entity: object, *, limit: int | None = None) -> _AsyncMessageIter:
        del entity
        assert limit is None
        return iterator

    client.iter_messages = MagicMock(side_effect=iter_messages)
    collector._client = client

    result = await collector._fetch_channel_messages("@big_chat", since=since)

    assert len(result) == 120
    assert iterator.consumed == 121


async def test_format_messages(tg_settings: SimpleNamespace):
    """Raw messages are formatted as readable markdown."""
    collector = TelegramChannelCollector(tg_settings)

    messages = {
        "@test_channel": _make_messages("Test Channel", ["Новость про AI", "Апдейт"]),
    }

    result = collector._format_messages(messages)

    assert "## Test Channel (2 сообщ.)" in result
    assert "Новость про AI" in result
    assert "Апдейт" in result
    assert "👁" in result  # views present


async def test_collect_and_save_writes_raw_messages(tg_settings: SimpleNamespace):
    """collect_and_save writes raw messages to inbox, no LLM calls."""
    collector = TelegramChannelCollector(tg_settings)

    messages = {
        "@test_channel": _make_messages("Test Channel", ["Первая новость"] * 3),
    }

    with patch.object(collector, "collect_messages", AsyncMock(return_value=messages)):
        summary = await collector.collect_and_save(episodic=None)

    assert "3 сообщений из 1 каналов" in summary

    inbox_file = list(Path(tg_settings.workspace_path).glob("inbox/channels_*.md"))
    assert len(inbox_file) == 1
    content = inbox_file[0].read_text()
    assert "Test Channel" in content
    assert "Первая новость" in content


async def test_handles_connection_failure(tg_settings: SimpleNamespace):
    """Connection failure is handled gracefully."""
    collector = TelegramChannelCollector(tg_settings)

    with patch.object(
        collector, "_create_client", side_effect=Exception("Connection failed")
    ):
        await collector.connect()

    # Should not raise, just log warning
    assert not collector._connected
