"""Tests for send_long_message utility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.utils import _split_text, send_long_message


@pytest.fixture
def mock_bot() -> MagicMock:
    bot = MagicMock()
    msg = MagicMock()
    msg.message_id = 1
    bot.send_message = AsyncMock(return_value=msg)
    return bot


@pytest.mark.asyncio
async def test_short_message_single_send(mock_bot: MagicMock) -> None:
    """Short text (<4096) sends as one message."""
    result = await send_long_message(mock_bot, 123, "Hello")
    assert len(result) == 1
    mock_bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text="Hello",
        parse_mode=None,
    )


@pytest.mark.asyncio
async def test_long_message_split(mock_bot: MagicMock) -> None:
    """Text >4096 chars is split into multiple messages."""
    text = "a" * 5000
    result = await send_long_message(mock_bot, 123, text, max_length=4096)
    assert mock_bot.send_message.await_count == 2
    assert len(result) == 2


def test_split_prefers_double_newline() -> None:
    """Split at \\n\\n over \\n when both available."""
    text = "A" * 40 + "\n\n" + "B" * 20 + "\n" + "C" * 40
    parts = _split_text(text, max_length=50)
    assert parts[0] == "A" * 40
    assert parts[1].startswith("B")


def test_split_prefers_newline_over_space() -> None:
    """Split at \\n over space when both available."""
    text = "A" * 40 + "\n" + "B" * 20 + " " + "C" * 40
    parts = _split_text(text, max_length=50)
    assert parts[0] == "A" * 40
    assert parts[1].startswith("B")


def test_split_no_whitespace_hard_cut() -> None:
    """Without whitespace, hard cut at max_length."""
    text = "X" * 100
    parts = _split_text(text, max_length=40)
    assert parts[0] == "X" * 40
    assert len(parts) == 3


def test_split_each_part_within_limit() -> None:
    """All chunks must be <= max_length."""
    text = "word " * 2000  # ~10000 chars
    parts = _split_text(text, max_length=4096)
    for part in parts:
        assert len(part) <= 4096


@pytest.mark.asyncio
async def test_send_with_parse_mode(mock_bot: MagicMock) -> None:
    """parse_mode is forwarded to send_message."""
    await send_long_message(mock_bot, 123, "hello", parse_mode="HTML")
    mock_bot.send_message.assert_awaited_once_with(
        chat_id=123,
        text="hello",
        parse_mode="HTML",
    )


def test_split_empty_text() -> None:
    """Empty text returns single empty part."""
    parts = _split_text("", max_length=4096)
    assert parts == [""]
