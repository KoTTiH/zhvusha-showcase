from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message  # noqa: TC002

from src.core.config import get_settings
from src.llm.protocols import LLMRequest
from src.llm.router import get_router
from src.skills.kwork_monitor.formatting import (
    build_draft_keyboard,
    build_evaluate_keyboard,
)
from src.skills.kwork_monitor.models import DraftState, ProjectCard
from src.skills.kwork_monitor.prompts import DRAFT_SYSTEM, DRAFT_USER, EVALUATE_USER

if TYPE_CHECKING:
    from src.skills.kwork_monitor.skill import KworkMonitorSkill

logger = structlog.get_logger()

router = Router(name="kwork_monitor")

CLEANUP_DELAY_SECONDS = 5 * 60  # 5 minutes

_cleanup_tasks: set[asyncio.Task[None]] = set()

_skill_instance: KworkMonitorSkill | None = None


def set_skill(skill: KworkMonitorSkill) -> None:
    """Wire the polling skill into the router so commands can act on it."""
    global _skill_instance
    _skill_instance = skill


def _admin_only(user_id: int | None) -> bool:
    if user_id is None:
        return False
    return user_id == get_settings().admin_user_id


@router.message(Command("kwork", "kwork_status"))
async def handle_kwork_status(message: Message) -> None:
    """Send current monitor status."""
    user_id = message.from_user.id if message.from_user else None
    if not _admin_only(user_id):
        return
    if _skill_instance is None or message.bot is None:
        return
    await _skill_instance.handle_status_command(
        bot=message.bot,
        chat_id=message.chat.id,
        command_message_id=message.message_id,
    )


@router.message(Command("sleep"))
async def handle_sleep(message: Message) -> None:
    """Pause monitoring for N hours (default 8)."""
    user_id = message.from_user.id if message.from_user else None
    if not _admin_only(user_id):
        return
    if _skill_instance is None:
        return

    text = message.text or ""
    parts = text.strip().split()
    try:
        hours = float(parts[1]) if len(parts) > 1 else 8
    except ValueError:
        hours = 8

    response = await _skill_instance.sleep(hours)
    await message.answer(response)


@router.message(Command("wake"))
async def handle_wake(message: Message) -> None:
    """Resume monitoring immediately."""
    user_id = message.from_user.id if message.from_user else None
    if not _admin_only(user_id):
        return
    if _skill_instance is None:
        return

    response = await _skill_instance.wake()
    await message.answer(response)


async def _safe_answer(
    callback: CallbackQuery,
    text: str | None = None,
    *,
    show_alert: bool = False,
) -> None:
    """Acknowledge a callback, swallowing 'query is too old' errors so the
    primary side-effect (edit/delete) still runs after a long downtime."""
    try:
        if show_alert:
            await callback.answer(text, show_alert=True)
        else:
            await callback.answer(text)
    except TelegramBadRequest as exc:
        logger.debug("callback_answer_failed", reason=str(exc))


def _schedule_delete(message: Message, delay: float = CLEANUP_DELAY_SECONDS) -> None:
    """Schedule message deletion after a delay."""

    async def _delete_later() -> None:
        await asyncio.sleep(delay)
        try:
            await message.delete()
        except Exception:
            logger.debug("message_delete_failed", message_id=message.message_id)

    task = asyncio.create_task(_delete_later())
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_tasks.discard)


# In-memory stores populated by skill (polling) and handlers (drafts)
_MAX_PROJECTS = 200
_projects: dict[int, ProjectCard] = {}
_drafts: dict[int, DraftState] = {}

# Episodic memory — set by KworkMonitorSkill.__init__
_episodic: Any = None


def set_episodic(ep: Any) -> None:
    """Set the episodic memory instance for kwork handlers."""
    global _episodic
    _episodic = ep


def register_project(card: ProjectCard) -> None:
    """Store a project card for later use by callback handlers."""
    if len(_projects) >= _MAX_PROJECTS:
        # Remove oldest entries (first inserted in dict order)
        excess = len(_projects) - _MAX_PROJECTS + 1
        for key in list(_projects)[:excess]:
            del _projects[key]
    _projects[card.id] = card


def get_project(project_id: int) -> ProjectCard | None:
    """Get a stored project card."""
    return _projects.get(project_id)


def get_draft(project_id: int) -> DraftState | None:
    """Get a stored draft."""
    return _drafts.get(project_id)


@router.callback_query(F.data.startswith("kwork:check:"))
async def handle_check(callback: CallbackQuery) -> None:
    """Evaluate a project before writing a draft."""
    project_id = int(callback.data.split(":")[-1])  # type: ignore[union-attr]
    card = _projects.get(project_id)

    if card is None:
        await _safe_answer(callback, "Проект не найден", show_alert=True)
        return

    await _safe_answer(callback, "Оцениваю проект...")

    router = get_router()
    prompt = EVALUATE_USER.format(
        title=card.title,
        description=card.description,
        price=card.price or "не указан",
        offers=card.offers or "?",
    )

    llm_response = await router.generate(
        LLMRequest(
            prompt=prompt,
            system=DRAFT_SYSTEM,
            tier="analyst",
            caller="kwork",
        )
    )
    verdict = llm_response.text

    keyboard = build_evaluate_keyboard(project_id)

    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"<b>Оценка: {card.title}</b>\n\n<blockquote expandable>{verdict}</blockquote>",
            reply_markup=keyboard,
            parse_mode="HTML",
        )

    logger.info("project_evaluated", project_id=project_id)


@router.callback_query(F.data.startswith("kwork:draft:"))
async def handle_draft(callback: CallbackQuery) -> None:
    """Generate a draft response for a project."""
    project_id = int(callback.data.split(":")[-1])  # type: ignore[union-attr]
    card = _projects.get(project_id)

    if card is None:
        await _safe_answer(callback, "Проект не найден", show_alert=True)
        return

    await _safe_answer(callback, "Генерирую отклик...")

    router = get_router()
    prompt = DRAFT_USER.format(
        title=card.title,
        description=card.description,
        price=card.price or "не указан",
    )

    draft_response = await router.generate(
        LLMRequest(
            prompt=prompt,
            system=DRAFT_SYSTEM,
            tier="analyst",
            caller="kwork",
        )
    )
    draft_text = draft_response.text

    _drafts[project_id] = DraftState(
        project_id=project_id,
        project_title=card.title,
        draft_text=draft_text,
    )

    keyboard = build_draft_keyboard(project_id)

    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"<b>Отклик на: {card.title}</b>\n\n<blockquote expandable>{draft_text}</blockquote>",
            reply_markup=keyboard,
            parse_mode="HTML",
        )

    logger.info("draft_generated", project_id=project_id, length=len(draft_text))


@router.callback_query(F.data.startswith("kwork:skip:"))
async def handle_skip(callback: CallbackQuery) -> None:
    """Skip a project — delete message."""
    await _safe_answer(callback, "Пропущено")
    if callback.message is not None:
        try:
            await callback.message.delete()  # type: ignore[union-attr]
        except Exception:
            logger.debug("skip_delete_failed", message_id=callback.message.message_id)


@router.callback_query(F.data.startswith("kwork:approve:"))
async def handle_approve(callback: CallbackQuery) -> None:
    """Send draft text as a copiable message."""
    project_id = int(callback.data.split(":")[-1])  # type: ignore[union-attr]
    draft = _drafts.pop(project_id, None)

    if draft is None:
        await _safe_answer(callback, "Черновик не найден", show_alert=True)
        return

    await _safe_answer(callback, "Готово!")

    if callback.message is not None:
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"📋 Отклик для копирования (удалится через 5 мин):\n\n<blockquote expandable>{draft.draft_text}</blockquote>",
            parse_mode="HTML",
        )
        _schedule_delete(callback.message)  # type: ignore[arg-type]

    logger.info("draft_approved", project_id=project_id)

    if _episodic is not None:
        from src.core.config import get_settings

        settings = get_settings()
        await _episodic.record(
            content=f"Nikita approved draft for: {draft.project_title}",
            user_id=settings.admin_user_id,
            chat_type="personal",
            role="user",
            source="kwork",
            importance=0.8,
            valence="positive",
            confidence=0.9,
            domain="kwork",
        )


@router.callback_query(F.data.startswith("kwork:edit:"))
async def handle_edit(callback: CallbackQuery) -> None:
    """Instruction to edit — FSM deferred to future."""
    await _safe_answer(
        callback,
        "Ответь реплаем на это сообщение с исправленным текстом (в разработке)",
        show_alert=True,
    )


@router.callback_query(F.data.startswith("kwork:cancel:"))
async def handle_cancel(callback: CallbackQuery) -> None:
    """Cancel draft — remove keyboard and clean up."""
    project_id = int(callback.data.split(":")[-1])  # type: ignore[union-attr]
    _drafts.pop(project_id, None)

    await _safe_answer(callback, "Отменено")
    if callback.message is not None:
        try:
            await callback.message.delete()  # type: ignore[union-attr]
        except Exception:
            logger.debug("cancel_delete_failed", message_id=callback.message.message_id)

    logger.info("draft_cancelled", project_id=project_id)

    if _episodic is not None:
        from src.core.config import get_settings

        settings = get_settings()
        await _episodic.record(
            content=f"Nikita cancelled draft for project {project_id}",
            user_id=settings.admin_user_id,
            chat_type="personal",
            role="user",
            source="kwork",
            importance=0.8,
            valence="negative",
            confidence=0.9,
            domain="kwork",
        )
