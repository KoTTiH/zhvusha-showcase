"""Redis Streams wrapper for daemon signal queues.

Three streams by priority:
- signals:critical — user messages, urgent deadlines
- signals:normal   — Kwork projects, scheduled events
- signals:background — channel posts, file changes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.daemon.signals import Signal, SignalPriority

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger()

WAKE_CHANNEL = "zhvusha:daemon:wake"

_STREAM_NAMES: dict[SignalPriority, str] = {
    "critical": "signals:critical",
    "normal": "signals:normal",
    "background": "signals:background",
}
_CONSUMER_GROUP = "zhvusha-daemon"
_MAXLEN = 10000


class SignalStream:
    """Priority-ordered Redis Streams consumer."""

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def ensure_groups(self) -> None:
        """Create consumer groups if they don't exist."""
        for stream_name in _STREAM_NAMES.values():
            try:
                await self._redis.xgroup_create(
                    stream_name, _CONSUMER_GROUP, id="0", mkstream=True
                )
            except Exception as exc:
                if "BUSYGROUP" not in str(exc):
                    logger.warning(
                        "xgroup_create_error", stream=stream_name, error=str(exc)
                    )

    async def push(self, signal: Signal) -> str:
        """Push a signal to the appropriate stream. Returns stream entry ID."""
        stream = _STREAM_NAMES[signal.priority]
        entry_id: Any = await self._redis.xadd(
            stream, signal.to_dict(), maxlen=_MAXLEN, approximate=True
        )
        logger.debug(
            "signal_pushed",
            stream=stream,
            signal_id=signal.id,
            entry_id=entry_id,
        )
        return str(entry_id)

    async def read_priority(
        self,
        consumer_name: str,
        count: int = 10,
        block_ms: int | None = None,
    ) -> list[Signal]:
        """Read signals in priority order: critical -> normal -> background.

        block_ms: None = don't block, positive int = block for N ms.
        Redis treats block=0 as "block forever", so we use None for non-blocking.
        """
        signals: list[Signal] = []

        for priority in ("critical", "normal", "background"):
            stream = _STREAM_NAMES[priority]
            # Only block on the first stream if explicitly requested
            block = block_ms if not signals else None
            try:
                kwargs: dict[str, Any] = {
                    "groupname": _CONSUMER_GROUP,
                    "consumername": consumer_name,
                    "streams": {stream: ">"},
                    "count": count - len(signals),
                }
                if block is not None:
                    kwargs["block"] = block
                entries: Any = await self._redis.xreadgroup(**kwargs)
            except Exception:
                logger.warning("stream_read_error", stream=stream, exc_info=True)
                continue

            if entries:
                for _stream_name, messages in entries:
                    for msg_id, data in messages:
                        sig = Signal.from_dict(
                            {
                                k.decode() if isinstance(k, bytes) else k: v.decode()
                                if isinstance(v, bytes)
                                else v
                                for k, v in data.items()
                            }
                        )
                        sig.stream_entry_id = msg_id
                        signals.append(sig)

            if len(signals) >= count:
                break

        return signals

    async def ack(self, signal: Signal) -> None:
        """Acknowledge a processed signal."""
        stream = _STREAM_NAMES[signal.priority]
        entry_id = signal.stream_entry_id
        if entry_id is not None:
            await self._redis.xack(stream, _CONSUMER_GROUP, entry_id)

    async def publish_wake(self) -> None:
        """Publish a wake signal via Redis Pub/Sub (cross-process)."""
        await self._redis.publish(WAKE_CHANNEL, "wake")

    async def start_wake_listener(self, on_wake: Callable[[], None]) -> None:
        """Subscribe to wake channel and invoke callback on each message.

        Runs forever — meant to be wrapped in asyncio.create_task and cancelled.
        """
        pubsub = self._redis.pubsub()
        try:
            await pubsub.subscribe(WAKE_CHANNEL)
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    on_wake()
        finally:
            try:
                await pubsub.unsubscribe(WAKE_CHANNEL)
            finally:
                await pubsub.aclose()
