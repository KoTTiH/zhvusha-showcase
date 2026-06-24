"""After the bot restarts, only Nikita's pending messages should be processed.

Telegram delivers everything queued during downtime on the next
``get_updates`` call. For a personal bot that means strangers get late
replies to messages they sent hours ago — which reads as if the bot
messaged them on restart. We drain pending updates before polling starts,
keep Nikita's updates in memory, and discard the rest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.bot.main import _drain_non_owner_pending


def _mk_update(
    update_id: int,
    *,
    user_id: int | None = None,
    kind: str = "message",
) -> MagicMock:
    update = MagicMock()
    update.update_id = update_id
    update.message = None
    update.edited_message = None
    update.callback_query = None
    update.channel_post = None
    if user_id is None:
        return update
    inner = MagicMock()
    inner.from_user = MagicMock()
    inner.from_user.id = user_id
    if kind == "message":
        update.message = inner
    elif kind == "edited_message":
        update.edited_message = inner
    elif kind == "callback_query":
        update.callback_query = inner
    elif kind == "channel_post":
        update.channel_post = inner
    return update


def _bot_with_updates(batches: list[list[Any]]) -> MagicMock:
    """Stub bot.get_updates to return given batches in order, then empty."""
    bot = MagicMock()
    calls: list[dict[str, Any]] = []
    batch_iter = iter(batches)

    async def get_updates(
        offset: int | None = None,
        timeout: int = 0,
        limit: int = 100,
    ) -> list[Any]:
        calls.append({"offset": offset, "timeout": timeout, "limit": limit})
        try:
            return next(batch_iter)
        except StopIteration:
            return []

    bot.get_updates = AsyncMock(side_effect=get_updates)
    bot._recorded_calls = calls  # type: ignore[attr-defined]
    return bot


async def test_empty_pending_returns_nothing() -> None:
    bot = _bot_with_updates([[]])
    owner = await _drain_non_owner_pending(bot, admin_user_id=42)
    assert owner == []


async def test_keeps_only_owner_messages_drops_strangers() -> None:
    batch = [
        _mk_update(10, user_id=42),
        _mk_update(11, user_id=999),
        _mk_update(12, user_id=42),
        _mk_update(13, user_id=123),
    ]
    bot = _bot_with_updates([batch])
    owner = await _drain_non_owner_pending(bot, admin_user_id=42)
    assert [u.update_id for u in owner] == [10, 12]


async def test_recognises_owner_in_callback_and_edited() -> None:
    batch = [
        _mk_update(1, user_id=42, kind="edited_message"),
        _mk_update(2, user_id=42, kind="callback_query"),
        _mk_update(3, user_id=999, kind="callback_query"),
    ]
    bot = _bot_with_updates([batch])
    owner = await _drain_non_owner_pending(bot, admin_user_id=42)
    assert sorted(u.update_id for u in owner) == [1, 2]


async def test_drops_updates_without_from_user() -> None:
    batch = [
        _mk_update(1, user_id=None),  # no sender attached (e.g. poll answer)
        _mk_update(2, user_id=42),
    ]
    bot = _bot_with_updates([batch])
    owner = await _drain_non_owner_pending(bot, admin_user_id=42)
    assert [u.update_id for u in owner] == [2]


async def test_advances_offset_until_empty_batch() -> None:
    first = [_mk_update(1, user_id=42), _mk_update(2, user_id=999)]
    second = [_mk_update(3, user_id=42)]
    bot = _bot_with_updates([first, second, []])

    owner = await _drain_non_owner_pending(bot, admin_user_id=42, limit=100)

    assert [u.update_id for u in owner] == [1, 3]
    calls = bot._recorded_calls
    # 1st call: no offset (initial). 2nd: offset=3 (after id 2). 3rd: offset=4.
    assert calls[0]["offset"] is None
    assert calls[1]["offset"] == 3
    assert calls[2]["offset"] == 4


async def test_final_ack_call_confirms_offset_even_if_last_batch_non_empty() -> None:
    """If the final batch was below the limit, we still need a final
    ``get_updates`` with the advanced offset so Telegram forgets the pending
    non-owner updates rather than re-delivering them when polling starts."""
    batch = [_mk_update(1, user_id=42), _mk_update(2, user_id=999)]
    bot = _bot_with_updates([batch, []])

    await _drain_non_owner_pending(bot, admin_user_id=42, limit=100)
    calls = bot._recorded_calls
    # Second call must carry offset past the stranger's update_id = 2.
    assert calls[-1]["offset"] == 3
