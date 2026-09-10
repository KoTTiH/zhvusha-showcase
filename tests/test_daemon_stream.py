"""Tests for daemon/stream.py — SignalStream Redis wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.daemon.signals import Signal
from src.daemon.stream import _CONSUMER_GROUP, _STREAM_NAMES, WAKE_CHANNEL, SignalStream


def _make_stream() -> tuple[SignalStream, AsyncMock]:
    redis = AsyncMock()
    return SignalStream(redis), redis


# --- ensure_groups ---


@pytest.mark.asyncio
async def test_ensure_groups_creates_all_streams() -> None:
    stream, redis = _make_stream()
    await stream.ensure_groups()
    assert redis.xgroup_create.await_count == 3
    for call_args in redis.xgroup_create.await_args_list:
        assert call_args.kwargs.get("mkstream") is True


@pytest.mark.asyncio
async def test_ensure_groups_ignores_busygroup() -> None:
    stream, redis = _make_stream()
    redis.xgroup_create = AsyncMock(side_effect=Exception("BUSYGROUP already exists"))
    await stream.ensure_groups()  # should not raise


@pytest.mark.asyncio
async def test_ensure_groups_logs_other_errors() -> None:
    stream, redis = _make_stream()
    redis.xgroup_create = AsyncMock(side_effect=Exception("connection refused"))
    await stream.ensure_groups()  # should not raise


# --- push ---


@pytest.mark.asyncio
async def test_push_normal_signal() -> None:
    stream, redis = _make_stream()
    redis.xadd = AsyncMock(return_value="1-0")

    signal = Signal(
        source="test",
        priority="normal",
        signal_type="test_event",
        payload={"key": "val"},
    )
    entry_id = await stream.push(signal)
    assert entry_id == "1-0"
    redis.xadd.assert_awaited_once()
    call_args = redis.xadd.call_args
    assert call_args[0][0] == _STREAM_NAMES["normal"]


@pytest.mark.asyncio
async def test_push_critical_signal_uses_critical_stream() -> None:
    stream, redis = _make_stream()
    redis.xadd = AsyncMock(return_value="2-0")

    signal = Signal(
        source="telegram",
        priority="critical",
        signal_type="user_message",
        payload={},
    )
    await stream.push(signal)
    assert redis.xadd.call_args[0][0] == _STREAM_NAMES["critical"]


# --- read_priority ---


@pytest.mark.asyncio
async def test_read_priority_empty() -> None:
    stream, redis = _make_stream()
    redis.xreadgroup = AsyncMock(return_value=[])
    signals = await stream.read_priority("consumer1")
    assert signals == []
    assert redis.xreadgroup.await_count == 3  # all 3 priorities tried


@pytest.mark.asyncio
async def test_read_priority_returns_signals() -> None:
    stream, redis = _make_stream()

    sig = Signal(source="test", priority="critical", signal_type="evt", payload={})
    msg_data = {
        k.encode(): v.encode() if isinstance(v, str) else str(v).encode()
        for k, v in sig.to_dict().items()
    }

    redis.xreadgroup = AsyncMock(
        side_effect=[
            [("signals:critical", [(b"1-0", msg_data)])],
            [],
            [],
        ]
    )

    signals = await stream.read_priority("consumer1")
    assert len(signals) >= 1


@pytest.mark.asyncio
async def test_read_priority_error_continues() -> None:
    stream, redis = _make_stream()
    redis.xreadgroup = AsyncMock(side_effect=Exception("connection lost"))
    signals = await stream.read_priority("consumer1")
    assert signals == []


# --- ack ---


@pytest.mark.asyncio
async def test_ack_calls_xack() -> None:
    stream, redis = _make_stream()
    signal = Signal(source="test", priority="normal", signal_type="evt", payload={})
    signal.stream_entry_id = b"1-0"

    await stream.ack(signal)
    redis.xack.assert_awaited_once_with(
        _STREAM_NAMES["normal"], _CONSUMER_GROUP, b"1-0"
    )


@pytest.mark.asyncio
async def test_ack_skips_when_no_entry_id() -> None:
    stream, redis = _make_stream()
    signal = Signal(source="test", priority="normal", signal_type="evt", payload={})
    signal.stream_entry_id = None

    await stream.ack(signal)
    redis.xack.assert_not_awaited()


# --- publish_wake ---


@pytest.mark.asyncio
async def test_publish_wake() -> None:
    stream, redis = _make_stream()
    await stream.publish_wake()
    redis.publish.assert_awaited_once_with(WAKE_CHANNEL, "wake")
