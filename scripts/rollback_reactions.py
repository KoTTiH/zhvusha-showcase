"""Blanket rollback of all reaction feedback events in the last 24 hours.

What it does:
    1. Finds every user episode in the admin's chat with valence != neutral
       (i.e. every message that triggered the learning loop in the last 24h)
    2. For each event:
        a. Resets the user episode importance/valence to defaults
        b. Finds the paired assistant episode (reply-target or latest-before)
           and resets its importance/valence to defaults
        c. Removes the mirror Telegram reaction from the bot's message
        d. Heuristically removes the reaction from the user's message,
           assuming its id = bot_message_id - 1 (usually holds in a 1-1 chat)

Defaults used when resetting:
    user episode:      importance=0.5, valence=neutral
    assistant episode: importance=0.3, valence=neutral

Usage:
    python scripts/rollback_reactions.py

No arguments — it wipes the full 24h window. Safe to re-run (idempotent for
already-reset episodes, Telegram API is graceful on no-op reaction clears).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from sqlalchemy import select, update
from src.core.config import get_settings
from src.memory.database import Episode, get_engine, get_session_maker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

DEFAULT_USER_IMPORTANCE = 0.5
DEFAULT_ASSISTANT_IMPORTANCE = 0.3
LOOKBACK_HOURS = 24
RECONSOLIDATION_WINDOW_HOURS = 6


async def _list_feedback_events(
    session_maker: async_sessionmaker,
    chat_id: int,
) -> list[Episode]:
    """Return user episodes flagged as feedback in the lookback window."""
    cutoff = datetime.now(tz=UTC) - timedelta(hours=LOOKBACK_HOURS)
    async with session_maker() as session:
        stmt = (
            select(Episode)
            .where(
                Episode.role == "user",
                Episode.chat_id == chat_id,
                Episode.timestamp >= cutoff,
                Episode.valence != "neutral",
            )
            .order_by(Episode.timestamp.asc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def _find_assistant_target(
    session_maker: async_sessionmaker,
    user_ep: Episode,
) -> Episode | None:
    """Find the assistant episode this feedback targeted.

    Priority:
        1. If user episode's metadata has reply_to_bot_message_id, look up
           that specific assistant episode by its bot_message_id.
        2. Otherwise, the latest assistant episode in the same chat within
           the reconsolidation window before this user episode.
    """
    reply_to_id: int | None = None
    if user_ep.metadata_json:
        try:
            meta = json.loads(user_ep.metadata_json)
            mid = meta.get("reply_to_bot_message_id")
            if isinstance(mid, int):
                reply_to_id = mid
        except (json.JSONDecodeError, AttributeError):
            pass

    cutoff = user_ep.timestamp - timedelta(hours=RECONSOLIDATION_WINDOW_HOURS)

    async with session_maker() as session:
        stmt = (
            select(Episode)
            .where(
                Episode.chat_id == user_ep.chat_id,
                Episode.role == "assistant",
                Episode.timestamp < user_ep.timestamp,
                Episode.timestamp >= cutoff,
            )
            .order_by(Episode.timestamp.desc())
        )
        result = await session.execute(stmt)
        candidates = list(result.scalars().all())

    if reply_to_id is not None:
        for ep in candidates:
            bot_id = _extract_bot_message_id(ep)
            if bot_id == reply_to_id:
                return ep

    return candidates[0] if candidates else None


def _extract_bot_message_id(ep: Episode) -> int | None:
    """Pull bot_message_id from an assistant episode's metadata_json."""
    if not ep.metadata_json:
        return None
    try:
        meta = json.loads(ep.metadata_json)
    except (json.JSONDecodeError, AttributeError):
        return None
    mid = meta.get("bot_message_id") if isinstance(meta, dict) else None
    return mid if isinstance(mid, int) else None


async def _reset_episode(
    session_maker: async_sessionmaker,
    episode_id: int,
    new_importance: float,
) -> None:
    """Reset one episode to (importance=given, valence=neutral)."""
    async with session_maker() as session:
        await session.execute(
            update(Episode)
            .where(Episode.id == episode_id)
            .values(importance=new_importance, valence="neutral")
        )
        await session.commit()


async def _remove_reaction(bot: Bot, chat_id: int, message_id: int, label: str) -> None:
    """Remove bot reaction from a message. Logs outcome, never raises."""
    try:
        await bot.set_message_reaction(
            chat_id=chat_id, message_id=message_id, reaction=[]
        )
        print(f"    tg reaction removed: {label} msg_id={message_id}")
    except TelegramAPIError as e:
        print(f"    tg remove failed: {label} msg_id={message_id} — {e}")


async def _rollback_event(
    session_maker: async_sessionmaker,
    bot: Bot,
    user_ep: Episode,
    chat_id: int,
) -> None:
    """Roll back a single feedback event: both episodes + both reactions."""
    u_preview = user_ep.content[:70].replace("\n", " ")
    ts = user_ep.timestamp.strftime("%H:%M:%S")
    print(
        f"\n[{ts}] user ep={user_ep.id} "
        f"imp={user_ep.importance:.2f} val={user_ep.valence}"
    )
    print(f"  USER: {u_preview}")

    target = await _find_assistant_target(session_maker, user_ep)
    if target is not None:
        t_preview = target.content[:70].replace("\n", " ")
        print(
            f"  → TARGET ep={target.id} "
            f"imp={target.importance:.2f} val={target.valence}"
        )
        print(f"  ZHV: {t_preview}")
    else:
        print("  → TARGET: none in 6h window")

    # Reset DB rows
    await _reset_episode(session_maker, user_ep.id, DEFAULT_USER_IMPORTANCE)
    print(f"    db reset: user ep={user_ep.id} → imp=0.5 val=neutral")

    if target is not None:
        await _reset_episode(session_maker, target.id, DEFAULT_ASSISTANT_IMPORTANCE)
        print(f"    db reset: target ep={target.id} → imp=0.3 val=neutral")

    # Remove Telegram reactions.
    # Mirror reaction on the bot message: we know its id exactly.
    # User reaction: we don't store user message_id in DB metadata, so we
    # sweep a window of message_ids below the bot message (bot_msg_id - 1..5).
    # set_message_reaction with reaction=[] is a safe no-op for messages
    # that don't have a bot reaction, so this oversweep is harmless.
    bot_msg_id = _extract_bot_message_id(target) if target else None
    if bot_msg_id is not None:
        await _remove_reaction(bot, chat_id, bot_msg_id, "mirror (bot)")
        for offset in range(1, 6):
            await _remove_reaction(
                bot, chat_id, bot_msg_id - offset, f"user sweep -{offset}"
            )
    else:
        print("    tg reaction skip: no bot_message_id in target metadata")


async def _sweep_all_reactions(
    session_maker: async_sessionmaker,
    bot: Bot,
    chat_id: int,
) -> None:
    """Fallback mode: try removing reactions from every assistant message
    in the last 24h, regardless of current valence.

    Needed when feedback events were already reset in the DB (first rollback
    pass neutralized valence), but Telegram reactions still appear in the UI.
    set_message_reaction with reaction=[] is a no-op for messages without a
    bot reaction, so this oversweep is safe.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(hours=LOOKBACK_HOURS)
    async with session_maker() as session:
        stmt = (
            select(Episode)
            .where(
                Episode.chat_id == chat_id,
                Episode.role == "assistant",
                Episode.timestamp >= cutoff,
            )
            .order_by(Episode.timestamp.asc())
        )
        result = await session.execute(stmt)
        assistants = list(result.scalars().all())

    bot_msg_ids = [mid for ep in assistants if (mid := _extract_bot_message_id(ep))]
    print(f"Sweep: found {len(bot_msg_ids)} bot messages with known msg_id.")

    for bot_msg_id in bot_msg_ids:
        await _remove_reaction(bot, chat_id, bot_msg_id, "sweep bot")
        for offset in range(1, 6):
            await _remove_reaction(
                bot, chat_id, bot_msg_id - offset, f"sweep user -{offset}"
            )


async def main() -> None:
    settings = get_settings()
    chat_id: int = settings.admin_user_id

    print(f"Rolling back all feedback events in last {LOOKBACK_HOURS}h")
    print(f"chat_id={chat_id}")
    print("=" * 72)

    engine = get_engine(settings.database_url)
    session_maker = get_session_maker(engine)

    events = await _list_feedback_events(session_maker, chat_id)
    bot = Bot(token=settings.bot_token)
    try:
        if events:
            print(f"Found {len(events)} feedback event(s) to roll back.")
            for user_ep in events:
                await _rollback_event(session_maker, bot, user_ep, chat_id)
        else:
            print("No active feedback events in DB (already reset).")

        # Always run a reaction sweep — catches stale reactions even after
        # DB was neutralized on a prior pass.
        print("\n" + "-" * 72)
        print("Sweeping all assistant messages for stale reactions...")
        print("-" * 72)
        await _sweep_all_reactions(session_maker, bot, chat_id)
    finally:
        await bot.session.close()
        await engine.dispose()

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
