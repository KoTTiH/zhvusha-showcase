"""Tests for ``chat_self_coding.events`` (Phase 40).

Block events are the bridge between the Editor / Architect cycles and
the chat-mode UI. The skill running inside the bot subscribes to a
per-user Redis channel; the cycles publish ``BlockEvent`` instances at
each transition (план готов / подготовка началась / реализация / готово
/ ошибка). Pub/Sub is fine — a single subscriber per user, no need for
consumer groups.

We test serialization round-trips (so the bot can decode whatever the
skill publishes), the no-op publisher (used by code paths that don't
yet wire a real publisher), and Redis-error swallowing (publishing
failures must not abort an Editor cycle).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock

import pytest


class FakeRedis:
    """In-memory ``redis.asyncio`` stand-in for ``publish``."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


# ---------------------------------------------------------------------------
# BlockEventType
# ---------------------------------------------------------------------------


class TestBlockEventType:
    def test_five_canonical_values(self) -> None:
        from src.skills.chat_self_coding.events import BlockEventType

        expected = {"plan", "preparation", "implementation", "done", "error"}
        assert {t.value for t in BlockEventType} == expected


# ---------------------------------------------------------------------------
# BlockEvent value type
# ---------------------------------------------------------------------------


class TestBlockEvent:
    def test_event_is_frozen(self) -> None:
        from src.skills.chat_self_coding.events import BlockEvent, BlockEventType

        evt = BlockEvent(
            user_id=1,
            event_type=BlockEventType.PLAN,
            slug="x",
            payload={},
        )
        with pytest.raises(FrozenInstanceError):
            evt.user_id = 2  # type: ignore[misc]

    def test_serialize_round_trip(self) -> None:
        """JSON has no tuple type — payloads use lists for cross-process
        consistency. Tuples in input become lists after round-trip, which
        is the documented contract."""
        from src.skills.chat_self_coding.events import BlockEvent, BlockEventType

        evt = BlockEvent(
            user_id=42,
            event_type=BlockEventType.DONE,
            slug="my-spec",
            task_id="code-task-fixed",
            payload={
                "description": "Расширила систему.",
                "files": ["a.py", "b.py"],
                "checks": [["тесты", True], ["стиль", True]],
            },
        )
        encoded = evt.serialize()
        decoded = BlockEvent.deserialize(encoded)
        assert decoded == evt
        assert decoded.task_id == "code-task-fixed"

    def test_serialize_is_json(self) -> None:
        """Encoded form must be a JSON string for human inspection."""
        import json

        from src.skills.chat_self_coding.events import BlockEvent, BlockEventType

        evt = BlockEvent(
            user_id=1,
            event_type=BlockEventType.PLAN,
            slug="x",
            payload={"summary": "y"},
        )
        decoded = json.loads(evt.serialize())
        assert decoded["user_id"] == 1
        assert decoded["event_type"] == "plan"
        assert decoded["slug"] == "x"
        assert decoded["task_id"] == ""
        assert decoded["payload"] == {"summary": "y"}

    def test_deserialize_rejects_unknown_event_type(self) -> None:
        import json

        from src.skills.chat_self_coding.events import BlockEvent

        bad = json.dumps(
            {
                "user_id": 1,
                "event_type": "not-a-real-type",
                "slug": "x",
                "payload": {},
            }
        )
        with pytest.raises(ValueError):
            BlockEvent.deserialize(bad)


# ---------------------------------------------------------------------------
# RedisBlockPublisher
# ---------------------------------------------------------------------------


class TestRedisBlockPublisher:
    async def test_publish_writes_to_per_user_channel(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            RedisBlockPublisher,
        )

        redis = FakeRedis()
        publisher = RedisBlockPublisher(redis=redis)
        await publisher.publish(
            BlockEvent(
                user_id=42,
                event_type=BlockEventType.PLAN,
                slug="x",
                payload={},
            )
        )
        assert len(redis.published) == 1
        channel, _ = redis.published[0]
        assert "42" in channel
        assert "chat_self_coding" in channel

    async def test_two_users_publish_to_separate_channels(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            RedisBlockPublisher,
        )

        redis = FakeRedis()
        publisher = RedisBlockPublisher(redis=redis)
        await publisher.publish(
            BlockEvent(user_id=1, event_type=BlockEventType.DONE, slug="a", payload={})
        )
        await publisher.publish(
            BlockEvent(user_id=2, event_type=BlockEventType.DONE, slug="b", payload={})
        )
        channels = {c for c, _ in redis.published}
        assert len(channels) == 2

    async def test_publish_includes_serialized_event(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            RedisBlockPublisher,
        )

        redis = FakeRedis()
        publisher = RedisBlockPublisher(redis=redis)
        evt = BlockEvent(
            user_id=1,
            event_type=BlockEventType.PLAN,
            slug="my-spec",
            payload={"summary": "Расширю..."},
        )
        await publisher.publish(evt)
        _, payload = redis.published[0]
        decoded = BlockEvent.deserialize(payload)
        assert decoded == evt

    async def test_publish_swallows_redis_errors(self) -> None:
        """A broken Redis must not crash the Editor cycle."""
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            RedisBlockPublisher,
        )

        broken = AsyncMock()
        broken.publish = AsyncMock(side_effect=ConnectionError("redis down"))
        publisher = RedisBlockPublisher(redis=broken)
        # Should not raise.
        await publisher.publish(
            BlockEvent(
                user_id=1,
                event_type=BlockEventType.ERROR,
                slug="x",
                payload={},
            )
        )

    async def test_channel_format_is_stable(self) -> None:
        """The bot listener and the publisher must agree on the channel
        name; pin the format here as a contract."""
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            RedisBlockPublisher,
            channel_for_user,
        )

        redis = FakeRedis()
        publisher = RedisBlockPublisher(redis=redis)
        await publisher.publish(
            BlockEvent(
                user_id=42,
                event_type=BlockEventType.PLAN,
                slug="x",
                payload={},
            )
        )
        assert redis.published[0][0] == channel_for_user(42)


# ---------------------------------------------------------------------------
# NoopBlockPublisher
# ---------------------------------------------------------------------------


class TestNoopPublisher:
    async def test_noop_does_nothing(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            NoopBlockPublisher,
        )

        publisher = NoopBlockPublisher()
        # No exception; nothing observable.
        await publisher.publish(
            BlockEvent(user_id=1, event_type=BlockEventType.PLAN, slug="x", payload={})
        )

    async def test_satisfies_protocol(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            BlockPublisher,
            NoopBlockPublisher,
        )

        publisher: BlockPublisher = NoopBlockPublisher()
        await publisher.publish(
            BlockEvent(user_id=1, event_type=BlockEventType.PLAN, slug="x", payload={})
        )


# ---------------------------------------------------------------------------
# Subscription helper (consumer side)
# ---------------------------------------------------------------------------


class TestSubscriptionHelper:
    """``subscribe_to_blocks`` returns an async iterator over decoded
    events. The bot listener uses this to render Telegram messages."""

    async def test_yields_decoded_events_from_pubsub(self) -> None:
        from src.skills.chat_self_coding.events import (
            BlockEvent,
            BlockEventType,
            subscribe_to_blocks,
        )

        # Hand-rolled fake pubsub object.
        evt = BlockEvent(
            user_id=42,
            event_type=BlockEventType.PLAN,
            slug="x",
            payload={"summary": "y"},
        )

        class FakePubSub:
            def __init__(self) -> None:
                self.subscribed: list[str] = []

            async def subscribe(self, channel: str) -> None:
                self.subscribed.append(channel)

            async def unsubscribe(self, channel: str) -> None:
                pass

            async def aclose(self) -> None:
                pass

            async def listen(self) -> Any:
                yield {"type": "subscribe", "data": b"1"}
                yield {"type": "message", "data": evt.serialize().encode("utf-8")}
                yield {"type": "message", "data": "stop"}

        pubsub = FakePubSub()

        class FakeRedisWithPubSub:
            def pubsub(self) -> FakePubSub:
                return pubsub

        events_seen: list[BlockEvent] = []
        async for blk in subscribe_to_blocks(redis=FakeRedisWithPubSub(), user_id=42):
            events_seen.append(blk)
            if len(events_seen) == 1:
                break

        assert events_seen == [evt]
        assert any("42" in c for c in pubsub.subscribed)
