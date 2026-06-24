"""Pinned Telegram message with live usage stats."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from aiogram.exceptions import TelegramBadRequest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from aiogram import Bot

    from src.monitoring.codex_limits import CodexLimitSnapshot
    from src.monitoring.usage_tracker import UsageTracker

logger = structlog.get_logger()

_MESSAGE_NOT_MODIFIED = "message is not modified"

_MONTH_NAMES_RU = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь",
}

_MONTH_NAMES_GEN_RU = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

DEFAULT_UPDATE_INTERVAL = 30  # seconds


def _format_model_line(today: Any, bucket: str) -> str | None:
    """Render one ``per-model`` row, or ``None`` to omit (zero calls)."""
    from src.llm.providers import get_pricing

    n = today.api_calls.get(bucket, 0)
    if not n:
        return None
    provider, _, model = bucket.partition("/")
    spec = get_pricing(provider, model) if provider else None
    m_in = today.input_tokens.get(bucket, 0)
    m_out = today.output_tokens.get(bucket, 0)
    if spec is not None:
        m_cost = (m_in * spec.input_per_mtok + m_out * spec.output_per_mtok) / 1_000_000
        emoji = spec.emoji
        # Prefer the shortest alias for display (haiku < sonnet < opus),
        # falling back to api_id if no aliases are declared.
        display_raw = spec.aliases[0] if spec.aliases else spec.api_id
    else:
        m_cost = 0.0
        emoji = "⚪"
        display_raw = model or bucket
    tok_str = f" ({m_in + m_out:,} tok)" if (m_in or m_out) else ""
    return f"  {emoji} {display_raw.capitalize()}: {n} × ${m_cost:.3f}{tok_str}"


class UsageDashboard:
    """Manages a pinned usage stats message in the admin chat."""

    def __init__(
        self,
        bot: Bot,
        admin_chat_id: int,
        tracker: UsageTracker,
        state_dir: Path,
        *,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        codex_limits_provider: Callable[[], CodexLimitSnapshot | None] | None = None,
    ) -> None:
        self._bot = bot
        self._chat_id = admin_chat_id
        self._tracker = tracker
        self._state_file = state_dir / "dashboard_state.json"
        self._update_interval = update_interval
        self._message_id: int | None = None
        self._last_update: float = 0.0
        self._pending: bool = False
        self._update_task: asyncio.Task[None] | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._codex_limits_provider = codex_limits_provider

    async def initialize(self) -> None:
        """Find or create the pinned dashboard message."""
        self._message_id = self._load_message_id()

        if self._message_id is not None:
            if not await self._try_edit():
                # Message was deleted — create new
                self._message_id = None
                await self._create_new_message()
            else:
                logger.info("dashboard_reused", message_id=self._message_id)
        else:
            await self._create_new_message()

        # Periodic flush — without this, ``_pending=True`` set inside the
        # rate-limit window would sit until the next caller happens to arrive.
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Cancel the background flush task — called at bot shutdown."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None

    async def _flush_loop(self) -> None:
        """Wake every ``update_interval`` seconds; flush if anything is pending."""
        interval = max(1, self._update_interval)
        try:
            while True:
                await asyncio.sleep(interval)
                if self._pending:
                    try:
                        await self.flush_pending()
                    except Exception:
                        logger.debug("dashboard_flush_loop_tick_failed", exc_info=True)
                elif self._codex_limits_provider is not None:
                    try:
                        self._last_update = time.monotonic()
                        await self._do_update()
                    except Exception:
                        logger.debug(
                            "dashboard_codex_limits_tick_failed",
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            raise

    def schedule_update(self) -> None:
        """Schedule a dashboard update (rate-limited).

        Safe to call from sync contexts: if no event loop is running, the
        rate-limit clock still advances but no task is spawned — the next
        async caller (``flush_pending`` or ``_flush_loop``) will deliver.
        """
        now = time.monotonic()
        if now - self._last_update < self._update_interval:
            self._pending = True
            return
        self._pending = False
        self._last_update = now
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._update_task = loop.create_task(self._do_update())

    async def flush_pending(self) -> None:
        """Force-send pending update if any."""
        if self._pending:
            self._pending = False
            self._last_update = time.monotonic()
            await self._do_update()

    async def _do_update(self) -> None:
        if self._message_id is None:
            return
        text = self._format_dashboard()
        try:
            await self._bot.edit_message_text(
                text=text,
                chat_id=self._chat_id,
                message_id=self._message_id,
                parse_mode="HTML",
            )
        except Exception as exc:
            if _is_message_not_modified(exc):
                logger.debug("dashboard_edit_noop", message_id=self._message_id)
                return
            logger.debug("dashboard_edit_failed", exc_info=True)

    async def _try_edit(self) -> bool:
        """Try to edit existing message. Returns True if success."""
        text = self._format_dashboard()
        try:
            await self._bot.edit_message_text(
                text=text,
                chat_id=self._chat_id,
                message_id=self._message_id,
                parse_mode="HTML",
            )
            return True
        except Exception as exc:
            if _is_message_not_modified(exc):
                return True
            logger.debug("dashboard_old_message_gone", exc_info=True)
            return False

    async def _create_new_message(self) -> None:
        text = self._format_dashboard()
        try:
            msg = await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
            self._message_id = msg.message_id
            self._save_message_id()

            await self._bot.pin_chat_message(
                chat_id=self._chat_id,
                message_id=msg.message_id,
                disable_notification=True,
            )
            logger.info("dashboard_created", message_id=msg.message_id)
        except Exception:
            logger.warning("dashboard_create_failed", exc_info=True)

    def _format_dashboard(self) -> str:
        now = datetime.now(tz=UTC)
        today = self._tracker.get_today()
        month_cost = self._tracker.get_month_total()

        day = now.day
        month_gen = _MONTH_NAMES_GEN_RU.get(now.month, "")
        month_name = _MONTH_NAMES_RU.get(now.month, "")
        time_str = now.strftime("%H:%M")

        lines: list[str] = []
        codex_limits = self._format_codex_limits()
        if codex_limits is not None:
            lines.append(f"🟣 {codex_limits}")
        lines.append(f"<b>📊 Сегодня {day} {month_gen}</b>")

        lines.extend(
            [
                "",
                f"<b>API:</b> {today.total_api} вызовов — ${today.cost_usd:.2f}",
            ]
        )

        all_buckets = sorted(
            {*today.api_calls, *today.input_tokens, *today.output_tokens}
        )
        breakdown = [
            line
            for bucket in all_buckets
            if (line := _format_model_line(today, bucket)) is not None
        ]
        lines.extend(breakdown if breakdown else ["  —"])

        # Cache metrics
        total_cache = today.cache_read_tokens + today.cache_write_tokens
        if total_cache > 0:
            hit_pct = today.cache_hit_rate * 100
            saved = today.cache_saved_usd
            lines.append(f"  💾 Cache: {hit_pct:.0f}% hit — -${saved:.2f}")

        # Subscription CLI
        lines.append("")
        cli_parts: list[str] = []
        if today.cli_sessions:
            cli_parts.append(f"{today.cli_sessions} сессия")
        if today.cli_calls:
            cli_parts.append(f"{today.cli_calls} вызовов")
        cli_label = ", ".join(cli_parts) if cli_parts else "—"
        lines.append(f"🟣 Codex CLI: {cli_label}")

        # Gemini
        gemini_label = f"{today.gemini_calls} (фото)" if today.gemini_calls else "—"
        lines.append(f"🟢 Gemini: {gemini_label}")

        # Per-function breakdown
        if today.caller_counts:
            lines.append("")
            lines.append("<b>По функциям:</b>")
            for caller, count in sorted(
                today.caller_counts.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {caller}: {count}")

        lines.append("")
        lines.append(f"📅 {month_name}: ${month_cost:.2f}")
        lines.append(f"<i>Обновлено: {time_str}</i>")

        return "\n".join(lines)

    def _format_codex_limits(self) -> str | None:
        if self._codex_limits_provider is None:
            return None
        try:
            snapshot = self._codex_limits_provider()
        except Exception:
            logger.debug("dashboard_codex_limits_failed", exc_info=True)
            return None
        if snapshot is None:
            return None
        from src.monitoring.codex_limits import format_codex_limit_snapshot

        return format_codex_limit_snapshot(snapshot)

    def _load_message_id(self) -> int | None:
        if not self._state_file.exists():
            return None
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            mid = data.get("message_id")
            return int(mid) if mid is not None else None
        except (json.JSONDecodeError, OSError):
            return None

    def _save_message_id(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._state_file.write_text(
                json.dumps({"message_id": self._message_id}), encoding="utf-8"
            )
        except OSError:
            logger.warning("dashboard_save_state_failed", exc_info=True)


def _is_message_not_modified(exc: Exception) -> bool:
    """Return True for Telegram's harmless identical-edit response."""
    message = str(exc).lower()
    if _MESSAGE_NOT_MODIFIED not in message:
        return False
    return isinstance(exc, TelegramBadRequest) or "bad request" in message
