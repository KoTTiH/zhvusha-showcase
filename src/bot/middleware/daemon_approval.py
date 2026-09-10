"""Middleware for daemon action approval via text replies.

When a user replies to a daemon approval notification,
this middleware intercepts the reply and updates the pending
action status in the database. Intent is classified by LLM (worker tier).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import BaseMiddleware

from src.daemon.pending_action import ActionStatus
from src.daemon.stream import WAKE_CHANNEL
from src.llm.protocols import LLMRequest

_ACTION_ID_RE = re.compile(r"#(\d+)")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Message, TelegramObject

    from src.daemon.pending_action import ApprovalStore, PendingActionDTO
    from src.llm.router import LLMRouter

logger = structlog.get_logger()

_CLASSIFY_SYSTEM = (
    "Ты классифицируешь ответ пользователя на pending daemon action. "
    "Пользователь может отвечать свободным текстом. Явное разрешение = approve, "
    "явный отказ/отмена = reject. Условия, правки, вопросы, переносы и "
    "сомнения = unclear, чтобы ответ ушёл обратно в диалог Жвуши. "
    "Ответь ОДНИМ словом: approve, reject или unclear."
)

_CLASSIFY_PROMPT = "Пользователь ответил на pending daemon action:\n\n{text}"


async def classify_intent(
    llm: LLMRouter, text: str, *, caller: str = "daemon_approval"
) -> str:
    """Classify user reply as approve/reject/unclear via LLM worker tier."""
    response = await llm.generate(
        LLMRequest(
            prompt=_CLASSIFY_PROMPT.format(text=text),
            system=_CLASSIFY_SYSTEM,
            tier="worker",
            temperature=0.0,
            caller=caller,
        )
    )
    token = response.text.strip().lower()
    if token in ("approve", "approved"):
        return "approve"
    if token in ("reject", "rejected"):
        return "reject"
    return "unclear"


class DaemonApprovalMiddleware(BaseMiddleware):
    """Intercept replies to daemon approval messages.

    Registered as inner middleware on dp.message — runs when a handler is
    matched, but passes through anything that is not a reply to an approval
    notification.
    """

    def __init__(
        self,
        approval_store: ApprovalStore,
        admin_user_id: int,
        llm_router: LLMRouter,
        redis: Any = None,
    ) -> None:
        super().__init__()
        self._store = approval_store
        self._admin_user_id = admin_user_id
        self._llm = llm_router
        self._redis = redis

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        msg: Message = event  # type: ignore[assignment]

        if msg.reply_to_message is None:
            return await handler(event, data)

        action = await self._find_action(msg)
        if action is None:
            return await handler(event, data)

        # Only admin can approve/reject daemon actions
        if not msg.from_user or msg.from_user.id != self._admin_user_id:
            return await handler(event, data)

        # This IS a reply to a daemon approval message — handle it
        if action.status != ActionStatus.PENDING:
            await msg.answer(f"Действие #{action.id} уже обработано ({action.status}).")
            return None

        text = (msg.text or "").strip()
        if not text:
            await msg.answer(
                "Не поняла решение. Напиши, разрешаешь действие, отклоняешь "
                "его или хочешь изменить условие."
            )
            return None

        try:
            intent = await classify_intent(self._llm, text)
        except Exception:
            logger.warning("classify_intent_failed", exc_info=True)
            await msg.answer("Ошибка при обработке ответа, попробуй ещё раз.")
            return None

        await self._apply_intent(msg, action, intent)
        return None

    async def _find_action(self, msg: Message) -> PendingActionDTO | None:
        """Look up the pending action for a replied message."""
        reply_msg_id: int = msg.reply_to_message.message_id  # type: ignore[union-attr]
        action = await self._store.get_by_telegram_message_id(reply_msg_id)
        if action is None:
            # Fallback: parse action_id from notification text (covers race
            # when telegram_message_id hasn't been persisted yet)
            replied_text = (
                msg.reply_to_message.text  # type: ignore[union-attr]
                or ""
            )
            match = _ACTION_ID_RE.search(replied_text)
            if match:
                action = await self._store.get_by_id(int(match.group(1)))
        return action

    async def _apply_intent(
        self, msg: Message, action: PendingActionDTO, intent: str
    ) -> None:
        """Apply classified intent to the pending action."""
        if intent == "approve":
            updated = await self._store.set_status(action.id, ActionStatus.APPROVED)
            if updated:
                await self._wake_daemon()
                await msg.answer(f"Действие #{action.id} одобрено. Выполню сейчас.")
                logger.info("daemon_action_approved", action_id=action.id)
            else:
                await msg.answer(f"Действие #{action.id} уже обработано.")
        elif intent == "reject":
            updated = await self._store.set_status(action.id, ActionStatus.REJECTED)
            if updated:
                await msg.answer(f"Действие #{action.id} отклонено.")
                logger.info("daemon_action_rejected", action_id=action.id)
            else:
                await msg.answer(f"Действие #{action.id} уже обработано.")
        else:
            await msg.answer(
                "Не поняла решение. Напиши, разрешаешь действие, отклоняешь "
                "его или хочешь изменить условие."
            )

    async def _wake_daemon(self) -> None:
        """Wake the daemon via Redis Pub/Sub so it picks up the approved action."""
        if self._redis is not None:
            try:
                await self._redis.publish(WAKE_CHANNEL, "wake")
            except Exception:
                logger.warning("daemon_wake_publish_failed", exc_info=True)
