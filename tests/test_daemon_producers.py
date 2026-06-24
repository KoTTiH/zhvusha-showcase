"""Tests for daemon/producers — signal emission to Redis Streams."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.daemon.producers.kwork_bridge import push_kwork_project
from src.daemon.producers.telegram import push_user_message
from src.daemon.stream import SignalStream

# --- kwork_bridge ---


@pytest.mark.asyncio
async def test_push_kwork_project() -> None:
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    stream = SignalStream(redis)

    await push_kwork_project(
        stream,
        project_id=123,
        title="Bot development",
        budget=5000,
        details={"category": "dev"},
    )
    redis.xadd.assert_awaited_once()
    call_data = redis.xadd.call_args[0][1]
    assert call_data["signal_type"] == "new_project"


@pytest.mark.asyncio
async def test_push_kwork_project_no_details() -> None:
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    stream = SignalStream(redis)

    await push_kwork_project(
        stream,
        project_id=456,
        title="Test",
        budget=3000,
    )
    redis.xadd.assert_awaited_once()


# --- telegram ---


@pytest.mark.asyncio
async def test_push_user_message() -> None:
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    stream = SignalStream(redis)

    await push_user_message(
        stream,
        user_id=12345,
        chat_id=67890,
        text="hello",
        message_id=1,
    )
    redis.xadd.assert_awaited_once()
    call_data = redis.xadd.call_args[0][1]
    assert call_data["signal_type"] == "user_message"
    assert call_data["priority"] == "critical"


# --- cron ---


@pytest.mark.asyncio
async def test_cron_producer_start_stop_without_apscheduler() -> None:
    """CronProducer gracefully handles missing apscheduler."""
    from src.daemon.producers.cron import CronProducer

    redis = AsyncMock()
    stream = SignalStream(redis)
    producer = CronProducer(stream)
    await producer.start()  # ImportError caught silently
    await producer.stop()  # No scheduler to stop


@pytest.mark.asyncio
async def test_cron_morning_briefing() -> None:
    """CronProducer._morning_briefing pushes a signal."""
    from src.daemon.producers.cron import CronProducer

    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    stream = SignalStream(redis)
    producer = CronProducer(stream)

    await producer._morning_briefing()
    redis.xadd.assert_awaited_once()


# --- filesystem ---


@pytest.mark.asyncio
async def test_filesystem_producer_start_no_paths() -> None:
    """FilesystemProducer with no watch_paths does nothing."""
    from src.daemon.producers.filesystem import FilesystemProducer

    redis = AsyncMock()
    stream = SignalStream(redis)
    ticker = AsyncMock()
    producer = FilesystemProducer(stream, ticker, watch_paths=[])
    await producer.start()
    assert producer._observer is None


@pytest.mark.asyncio
async def test_filesystem_producer_stop_no_observer() -> None:
    """Stop with no observer is a no-op."""
    from src.daemon.producers.filesystem import FilesystemProducer

    redis = AsyncMock()
    stream = SignalStream(redis)
    ticker = AsyncMock()
    producer = FilesystemProducer(stream, ticker)
    await producer.stop()  # no observer, should not raise
