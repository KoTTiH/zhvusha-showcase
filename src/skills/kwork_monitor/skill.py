"""Kwork monitor — background skill that polls the Kwork exchange.

v4 ``BackgroundSkill`` (phase 7.2). New instances start in sleep mode by
default and run an infinite polling loop after :meth:`start_polling` in the bot
startup hook. The legacy ``/kwork`` / ``/sleep`` / ``/wake`` /
``/kwork_status`` handlers remain in ``handlers.py``; chat-first control goes
through ``can_handle`` / ``execute`` so SkillInvocationService stays the gate.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from decimal import Decimal
from typing import Any, ClassVar, Literal

import structlog

from src.core.config import get_settings
from src.skills.base import (
    AgentContext,
    BackgroundSkill,
    ExecutionPlan,
    SideEffect,
    SkillResult,
)
from src.skills.kwork_monitor.filters import filter_projects
from src.skills.kwork_monitor.formatting import (
    build_project_keyboard,
    format_project_card,
)
from src.skills.kwork_monitor.handlers import register_project, set_episodic

logger = structlog.get_logger()

SEEN_KEY = "kwork:seen"
SEEN_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
DEFAULT_SLEEP_HOURS = 8
_SENSITIVE_ERROR_FIELDS = (
    "login",
    "password",
    "phone_last",
    "token",
    "authorization",
    "api_key",
    "kwork_login",
    "kwork_password",
    "kwork_phone_last",
)
_SENSITIVE_FIELD_ALTERNATION = "|".join(
    re.escape(field) for field in _SENSITIVE_ERROR_FIELDS
)
_QUOTED_KEY_VALUE_RE = re.compile(
    rf"(?i)(?P<prefix>['\"](?:{_SENSITIVE_FIELD_ALTERNATION})['\"]\s*[:=]\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
)
_QUOTED_VALUE_RE = re.compile(
    rf"(?i)(?P<prefix>\b(?:{_SENSITIVE_FIELD_ALTERNATION})\b\s*[:=]\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
)
_UNQUOTED_VALUE_RE = re.compile(
    rf"(?i)(?P<prefix>\b(?:{_SENSITIVE_FIELD_ALTERNATION})\b\s*[:=]\s*)"
    r"(?P<value>[^,\s)}\]]+)"
)
_BASIC_AUTH_RE = re.compile(r"(?i)\b(Basic\s+)[A-Za-z0-9._~+/=-]+")
_BEARER_AUTH_RE = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+")
_MAX_ERROR_LOG_CHARS = 500
_KWORK_STATUS_PREFIXES: tuple[str, ...] = (
    "покажи свежие kwork",
    "покажи свежие кворк",
    "покажи kwork",
    "покажи кворк",
    "статус kwork",
    "статус кворк",
    "что по kwork",
    "что по кворк",
    "свежие kwork",
    "свежие кворк",
)
_KWORK_SLEEP_PREFIXES: tuple[str, ...] = (
    "усыпи мониторинг",
    "усыпи kwork",
    "усыпи кворк",
    "приостанови мониторинг",
    "приостанови kwork",
    "приостанови кворк",
)
_KWORK_WAKE_PREFIXES: tuple[str, ...] = (
    "разбуди мониторинг",
    "разбуди kwork",
    "разбуди кворк",
    "возобнови мониторинг",
    "возобнови kwork",
    "возобнови кворк",
)
_HOURS_RE = re.compile(r"\b(?P<hours>\d+(?:[.,]\d+)?)\b")


def _sleep_deadline(hours: float = DEFAULT_SLEEP_HOURS) -> float:
    return time.monotonic() + hours * 3600


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _matches_route_prefix(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)


def _chat_action(message: str) -> Literal["status", "sleep", "wake"] | None:
    text = _normalize_chat_route_text(message)
    if text in {"/kwork", "/kwork_status"} or text.startswith("/kwork "):
        return "status"
    if text == "/sleep" or text.startswith("/sleep "):
        return "sleep"
    if text == "/wake" or text.startswith("/wake "):
        return "wake"
    if _matches_route_prefix(text, _KWORK_STATUS_PREFIXES):
        return "status"
    if _matches_route_prefix(text, _KWORK_SLEEP_PREFIXES):
        return "sleep"
    if _matches_route_prefix(text, _KWORK_WAKE_PREFIXES):
        return "wake"
    return None


def _parse_sleep_hours(message: str) -> float:
    match = _HOURS_RE.search(message.replace(",", "."))
    if match is None:
        return float(DEFAULT_SLEEP_HOURS)
    try:
        parsed = float(match.group("hours"))
    except ValueError:
        return float(DEFAULT_SLEEP_HOURS)
    return min(max(parsed, 0.25), 24.0)


def _sanitize_kwork_error_text(text: str) -> str:
    """Remove credentials from exception text before it reaches structlog."""
    redacted = _BASIC_AUTH_RE.sub(r"\1***", text)
    redacted = _BEARER_AUTH_RE.sub(r"\1***", redacted)
    redacted = _QUOTED_KEY_VALUE_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}***{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _QUOTED_VALUE_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}***{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _UNQUOTED_VALUE_RE.sub(
        lambda match: f"{match.group('prefix')}***",
        redacted,
    )
    if len(redacted) <= _MAX_ERROR_LOG_CHARS:
        return redacted
    return f"{redacted[:_MAX_ERROR_LOG_CHARS]}...[truncated]"


class KworkMonitorSkill(BackgroundSkill):
    """Polls the Kwork exchange on a timer and sends cards to the admin chat."""

    name: ClassVar[str] = "kwork_monitor"
    description: ClassVar[str] = "Monitors Kwork project exchange for relevant projects"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "analyst"

    triggers: ClassVar[list[str]] = ["/kwork", "/kwork_status", "/sleep", "/wake"]

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "medium"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FROM_KB,
        SideEffect.WRITES_TO_KB,
        SideEffect.CALLS_LLM,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.NETWORK_IO_EXTERNAL,
        SideEffect.MODIFIES_MEMORY,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    trigger_type: ClassVar[Literal["cron", "event", "interval"]] = "interval"
    trigger_config: ClassVar[dict[str, Any]] = {"interval_seconds": 600}

    def __init__(self, episodic: Any = None) -> None:
        self._task: asyncio.Task[None] | None = None
        self._redis: Any = None
        self._kwork_client: Any = None
        self._poll_count: int = 0
        self._last_status_msg_id: int | None = None
        self._last_command_msg_id: int | None = None
        self._status_chat_id: int | None = None
        self._sleeping_until: float | None = _sleep_deadline()
        self._wake_task: asyncio.Task[None] | None = None
        self._episodic = episodic
        if episodic is not None:
            set_episodic(episodic)

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal":
            return 0.0
        return 0.93 if _chat_action(message) is not None else 0.0

    def requires_approval_for_message(
        self,
        message: str,
        context: AgentContext,
    ) -> bool:
        action = _chat_action(message)
        return context.mode == "personal" and action in {"sleep", "wake"}

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        action = _chat_action(message) or "status"
        summary_by_action = {
            "status": "Показать статус мониторинга Kwork.",
            "sleep": (
                f"Приостановить мониторинг Kwork на {_parse_sleep_hours(message):g} ч."
            ),
            "wake": "Возобновить мониторинг Kwork.",
        }
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="background",
            human_summary=summary_by_action[action],
            estimated_tokens=0,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=1.0,
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=0,
            metadata={"kwork_action": action},
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Handle explicit chat-first controls for the background monitor."""
        if context.mode != "personal":
            return SkillResult(success=False, response="")
        action = _chat_action(message)
        if action == "status":
            return SkillResult(
                success=True,
                response=self.format_status(),
                metadata={"skill_name": self.name, "kwork_action": "status"},
            )
        if action == "sleep":
            response = await self.sleep(_parse_sleep_hours(message))
            return SkillResult(
                success=True,
                response=response,
                metadata={"skill_name": self.name, "kwork_action": "sleep"},
            )
        if action == "wake":
            response = await self.wake()
            return SkillResult(
                success=True,
                response=response,
                metadata={"skill_name": self.name, "kwork_action": "wake"},
            )
        return SkillResult(success=False, response="")

    async def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def _is_seen(self, project_id: int) -> bool:
        r = await self._get_redis()
        return bool(await r.sismember(SEEN_KEY, str(project_id)))

    async def _mark_seen(self, project_id: int) -> None:
        r = await self._get_redis()
        await r.sadd(SEEN_KEY, str(project_id))
        await r.expire(SEEN_KEY, SEEN_TTL_SECONDS)

    async def _get_seen_ids(self) -> set[int]:
        r = await self._get_redis()
        members = await r.smembers(SEEN_KEY)
        return {int(m) for m in members}

    async def _get_kwork_client(self) -> Any:
        if self._kwork_client is None:
            from kwork import KworkClient

            settings = get_settings()
            self._kwork_client = KworkClient(
                login=settings.kwork_login,
                password=settings.kwork_password,
                phone_last=settings.kwork_phone_last or None,
                relogin_on_auth_error=True,
            )
        return self._kwork_client

    async def start_polling(self, bot: Any) -> None:
        """Start the background polling task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(bot))
        logger.info("kwork_monitor_started")

    async def stop_polling(self) -> None:
        """Stop the background polling task and close resources."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        if self._wake_task is not None:
            self._wake_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._wake_task
            self._wake_task = None

        self._sleeping_until = None

        if self._kwork_client is not None:
            await self._kwork_client.close()
            self._kwork_client = None

        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

        logger.info("kwork_monitor_stopped")

    async def _poll_loop(self, bot: Any) -> None:  # pragma: no cover
        """Infinite polling loop."""
        settings = get_settings()
        while True:
            try:
                await self._poll_once(bot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "kwork_poll_error",
                    error_type=type(exc).__name__,
                    error=_sanitize_kwork_error_text(str(exc)),
                )
            await asyncio.sleep(settings.kwork_poll_interval_seconds)

    @property
    def is_sleeping(self) -> bool:
        """Check if monitor is currently in sleep mode."""
        if self._sleeping_until is None:
            return False
        if time.monotonic() >= self._sleeping_until:
            self._sleeping_until = None
            return False
        return True

    def sleep_remaining_minutes(self) -> int:
        """Minutes left until auto-wake."""
        if self._sleeping_until is None:
            return 0
        remaining = self._sleeping_until - time.monotonic()
        return max(0, int(remaining / 60))

    async def sleep(self, hours: float = DEFAULT_SLEEP_HOURS) -> str:
        """Pause monitoring for N hours."""
        self._sleeping_until = _sleep_deadline(hours)

        if self._wake_task is not None:
            self._wake_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._wake_task

        async def _auto_wake() -> None:
            await asyncio.sleep(hours * 3600)
            self._sleeping_until = None
            logger.info("kwork_monitor_auto_wake")

        self._wake_task = asyncio.create_task(_auto_wake())

        h = int(hours)
        m = int((hours - h) * 60)
        duration = f"{h}ч" if m == 0 else f"{h}ч {m}мин"
        logger.info("kwork_monitor_sleep", hours=hours)
        return f"💤 Мониторинг приостановлен на {duration}. /wake чтобы разбудить"

    async def wake(self) -> str:
        """Resume monitoring immediately."""
        was_sleeping = self.is_sleeping
        self._sleeping_until = None

        if self._wake_task is not None:
            self._wake_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._wake_task
            self._wake_task = None

        if was_sleeping:
            logger.info("kwork_monitor_wake")
            return "☀️ Мониторинг возобновлён"
        return "Мониторинг и так активен"

    async def _poll_once(self, bot: Any) -> None:
        """Single polling cycle: fetch, filter, notify."""
        if self.is_sleeping:
            logger.debug("kwork_poll_skipped_sleeping")
            return

        settings = get_settings()
        keywords = [
            kw.strip() for kw in settings.kwork_keywords.split(",") if kw.strip()
        ]
        seen_ids = await self._get_seen_ids()

        client = await self._get_kwork_client()
        raw_projects = await client.get_projects(categories_ids=["all"])

        cards = filter_projects(
            raw_projects,
            keywords=keywords,
            min_budget=settings.kwork_min_budget,
            max_offers=settings.kwork_max_offers,
            seen_ids=seen_ids,
        )

        for card in cards:
            text = format_project_card(card)
            keyboard = build_project_keyboard(card.id)

            await bot.send_message(
                chat_id=settings.admin_user_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            register_project(card)
            await self._mark_seen(card.id)

            if self._episodic is not None:
                await self._episodic.record(
                    content=f"Found Kwork project: {card.title}, budget {card.price}",
                    user_id=settings.admin_user_id,
                    chat_type="personal",
                    role="assistant",
                    source="kwork",
                    importance=0.6,
                    domain="kwork",
                )

        self._poll_count += 1
        logger.info(
            "kwork_poll_complete",
            found=len(cards),
            total_raw=len(raw_projects),
            poll_number=self._poll_count,
        )

    async def _delete_old_messages(self, bot: Any) -> None:
        """Delete previous command + status messages."""
        if self._status_chat_id is None:
            return
        for msg_id in (self._last_command_msg_id, self._last_status_msg_id):
            if msg_id is not None:
                try:
                    await bot.delete_message(
                        chat_id=self._status_chat_id, message_id=msg_id
                    )
                except Exception:
                    logger.debug("old_message_delete_failed", message_id=msg_id)
        self._last_command_msg_id = None
        self._last_status_msg_id = None

    async def handle_status_command(
        self, bot: Any, chat_id: int, command_message_id: int | None
    ) -> None:
        """Handle ``/kwork`` / ``/kwork_status`` — cleanup + send current status."""
        await self._delete_old_messages(bot)
        sent = await bot.send_message(chat_id=chat_id, text=self.format_status())
        self._last_command_msg_id = command_message_id
        self._last_status_msg_id = sent.message_id
        self._status_chat_id = chat_id

    def format_status(self) -> str:
        """Build status text."""
        is_running = self._task is not None and not self._task.done()
        if self.is_sleeping:
            remaining = self.sleep_remaining_minutes()
            return (
                f"💤 Мониторинг спит (проснётся через {remaining} мин)\n"
                f"Циклов опроса: {self._poll_count}"
            )
        status = "активен" if is_running else "остановлен"
        return f"Kwork мониторинг: {status}\nЦиклов опроса: {self._poll_count}"
