"""Block events — bridge between cycle workers and the chat-mode UI (Phase 40).

The ``ideation_to_spec`` and ``implement_spec`` cycles are long-running:
they don't return per-stage results to a single caller, they emit
``BlockEvent``s as they cross transitions. The chat-mode skill running
inside the bot subscribes to a per-user Redis Pub/Sub channel and turns
those events into Telegram block messages (see ``blocks.py``).

Why Pub/Sub rather than Streams: there is exactly one consumer per
user (the bot process), no replay requirement, and a missed event during
a bot restart is acceptable — the user can always ask «как идёт» to
re-pull state from the spec yaml. Pub/Sub keeps the moving parts down
and avoids consumer-group bookkeeping.

Publishing is best-effort: a Redis outage must not abort an Editor
cycle, so ``RedisBlockPublisher.publish`` swallows exceptions after
logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------


class BlockEventType(StrEnum):
    """Five canonical event kinds — one per chat-mode block type."""

    PLAN = "plan"
    PREPARATION = "preparation"
    IMPLEMENTATION = "implementation"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class BlockEvent:
    """Immutable event published by cycle workers, consumed by the bot.

    ``payload`` is event-type-specific:

    * ``PLAN``: ``{"summary": str, "files": list[str], "tier": int}``
    * ``PREPARATION``: ``{}`` (static block — no data)
    * ``IMPLEMENTATION``: ``{}``
    * ``DONE``: ``{"description": str, "files": list[str], "checks": list[[str, bool]]}``
    * ``ERROR``: ``{"reason": str, "next_step": str}``

    The skill that consumes the event is responsible for projecting the
    payload onto the matching ``blocks.*Block`` value type.
    """

    user_id: int
    event_type: BlockEventType
    slug: str
    payload: dict[str, Any]
    task_id: str = ""

    def serialize(self) -> str:
        return json.dumps(
            {
                "user_id": self.user_id,
                "event_type": self.event_type.value,
                "slug": self.slug,
                "task_id": self.task_id,
                "payload": self.payload,
            }
        )

    @classmethod
    def deserialize(cls, raw: str | bytes) -> BlockEvent:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        try:
            event_type = BlockEventType(data["event_type"])
        except ValueError as exc:
            raise ValueError(
                f"Unknown BlockEventType: {data.get('event_type')!r}"
            ) from exc
        return cls(
            user_id=int(data["user_id"]),
            event_type=event_type,
            slug=str(data["slug"]),
            payload=dict(data.get("payload", {})),
            task_id=str(data.get("task_id", "")),
        )


# ---------------------------------------------------------------------------
# Channel naming
# ---------------------------------------------------------------------------

_CHANNEL_PREFIX = "chat_self_coding:blocks:"


def channel_for_user(user_id: int) -> str:
    """Per-user Redis Pub/Sub channel name.

    Stable contract — both publisher and subscriber must agree.
    """
    return f"{_CHANNEL_PREFIX}{user_id}"


# ---------------------------------------------------------------------------
# Publisher contract + implementations
# ---------------------------------------------------------------------------


class BlockPublisher(Protocol):
    """Async publisher for chat-mode block events."""

    async def publish(self, event: BlockEvent) -> None: ...


class RedisBlockPublisher:
    """Publishes events via Redis Pub/Sub on a per-user channel.

    ``redis`` must implement ``async def publish(channel, message)`` —
    the standard ``redis.asyncio.Redis`` shape. Failures are logged and
    swallowed; cycles stay healthy even when Redis is unhealthy.
    """

    def __init__(self, *, redis: Any) -> None:
        self._redis = redis

    async def publish(self, event: BlockEvent) -> None:
        channel = channel_for_user(event.user_id)
        try:
            await self._redis.publish(channel, event.serialize())
        except Exception:
            logger.warning(
                "block_event_publish_failed",
                channel=channel,
                event_type=event.event_type.value,
                exc_info=True,
            )


class NoopBlockPublisher:
    """No-op publisher.

    Used as the default in cycle workers that aren't wired to chat mode
    yet, and in tests that don't care about the publish side. Lets call
    sites depend on the ``BlockPublisher`` Protocol without a Redis.
    """

    async def publish(self, event: BlockEvent) -> None:
        del event


# ---------------------------------------------------------------------------
# Subscriber helper
# ---------------------------------------------------------------------------


async def subscribe_to_blocks(*, redis: Any, user_id: int) -> AsyncIterator[BlockEvent]:
    """Async iterator yielding decoded ``BlockEvent``s for a single user.

    The bot wires this into a long-running task; on each iteration it
    renders the event as a Telegram block message. Non-message Pub/Sub
    frames (subscribe ack, etc.) are skipped silently. Decoding errors
    are logged and skipped — one bad payload must not kill the listener.
    """
    pubsub = redis.pubsub()
    channel = channel_for_user(user_id)
    try:
        await pubsub.subscribe(channel)
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            data = msg.get("data")
            if data is None:
                continue
            try:
                yield BlockEvent.deserialize(data)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                logger.warning(
                    "block_event_decode_failed",
                    channel=channel,
                    raw=str(data)[:200],
                    exc_info=True,
                )
                continue
    finally:
        try:
            await pubsub.unsubscribe(channel)
        finally:
            await pubsub.aclose()
