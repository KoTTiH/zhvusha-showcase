from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import Router
from aiogram.filters import Command

from src.bot.maintenance_guard import (
    MaintenanceGuardError,
    MorningMaintenanceGuard,
    default_morning_maintenance_marker_path,
)
from src.core.config import get_settings
from src.memory import ConsolidationLock
from src.skills.base import AgentContext
from src.skills.workspace_session.workspace import get_workspace_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from aiogram.types import Message

    from src.skills.base import BaseSkill
    from src.skills.invocation import SkillInvocationService
    from src.skills.workspace_session.skill import WorkspaceSessionSkill

logger = structlog.get_logger()

router = Router(name="morning")

_FALLBACK_LOOKBACK_HOURS = 24
_MIN_LOOKBACK_HOURS = 1
_MAX_LOOKBACK_HOURS = 720

_skill: WorkspaceSessionSkill | None = None
_invocation_service: SkillInvocationService | None = None
_invocation_skills: Sequence[BaseSkill] = ()


def set_skill(skill: WorkspaceSessionSkill | None) -> None:
    """Inject skill instance (called from main.py on startup)."""
    global _skill
    _skill = skill


def set_invocation_service(
    service: SkillInvocationService | None,
    skills: Sequence[BaseSkill] = (),
) -> None:
    """Inject the central skill gate used by the /morning production handler."""
    global _invocation_service, _invocation_skills
    _invocation_service = service
    _invocation_skills = skills


async def _default_lookback_hours(settings: object) -> int:
    raw_workspace = getattr(settings, "workspace_path", "~/zhvusha-workspace")
    workspace_path = get_workspace_path(str(raw_workspace or "~/zhvusha-workspace"))
    lock = ConsolidationLock(workspace_path / "personality")
    last_consolidated_at = await lock.read_last_consolidated_at()
    source = "marker"
    if last_consolidated_at <= 0:
        last_consolidated_at = _legacy_consolidation_marker_mtime(workspace_path)
        source = "legacy_consolidation_results"
    if last_consolidated_at <= 0:
        logger.info(
            "morning_lookback_defaulted",
            hours=_MAX_LOOKBACK_HOURS,
            reason="no_consolidation_marker",
        )
        return _MAX_LOOKBACK_HOURS
    elapsed_hours = math.ceil(max(0.0, time.time() - last_consolidated_at) / 3600)
    hours = min(_MAX_LOOKBACK_HOURS, max(_MIN_LOOKBACK_HOURS, elapsed_hours))
    logger.info("morning_lookback_computed", hours=hours, source=source)
    return hours


def _legacy_consolidation_marker_mtime(workspace_path: Path) -> float:
    """Infer the previous consolidation time for workspaces without the marker."""
    candidates = [
        workspace_path / "inbox" / ".processed" / "consolidation_results.md",
        workspace_path / "inbox" / "consolidation_results.md",
    ]
    mtimes: list[float] = []
    for path in candidates:
        try:
            if path.is_file():
                mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=0.0)


def _parse_explicit_lookback_hours(text: str) -> tuple[int | None, str | None]:
    parts = text.split()
    if len(parts) <= 1:
        return None, None
    try:
        hours = int(parts[1])
    except ValueError:
        return None, "Укажи число часов: /morning 48"
    if hours < 1:
        return None, "Минимум 1 час."
    if hours > 720:
        return None, "Максимум 720 часов (30 дней)."
    return hours, None


def _build_morning_maintenance_guard(
    settings: object,
) -> MorningMaintenanceGuard | None:
    if not getattr(settings, "autonomous_self_coding_morning_guard_enabled", False):
        return None
    raw_workspace = getattr(settings, "workspace_path", "~/zhvusha-workspace")
    workspace_path = get_workspace_path(str(raw_workspace or "~/zhvusha-workspace"))
    return MorningMaintenanceGuard(
        marker_path=default_morning_maintenance_marker_path(workspace_path),
    )


async def _invoke_morning_workspace_session(
    *,
    message: Message,
    context: AgentContext,
    hours: int,
    maintenance_guard: MorningMaintenanceGuard | None,
) -> Any:
    if _invocation_service is None or _skill is None:
        raise RuntimeError("Morning invocation dependencies are not configured")
    if maintenance_guard is None:
        return await _invocation_service.invoke_named_skill(
            "/morning",
            context,
            _invocation_skills,
            _skill.name,
        )
    with maintenance_guard.active(
        owner="morning",
        metadata={
            "chat_id": message.chat.id,
            "message_id": message.message_id,
            "lookback_hours": hours,
        },
    ):
        return await _invocation_service.invoke_named_skill(
            "/morning",
            context,
            _invocation_skills,
            _skill.name,
        )


async def _invoke_morning_or_report_guard_error(
    *,
    message: Message,
    context: AgentContext,
    hours: int,
    maintenance_guard: MorningMaintenanceGuard | None,
) -> Any | None:
    try:
        return await _invoke_morning_workspace_session(
            message=message,
            context=context,
            hours=hours,
            maintenance_guard=maintenance_guard,
        )
    except MaintenanceGuardError:
        logger.warning("morning_maintenance_guard_failed", exc_info=True)
        await message.answer(
            "Не могу безопасно запустить /morning: маркер окна восстановления не записался."
        )
        return None


@router.message(Command("morning"))
async def handle_morning(message: Message, mode: str = "personal") -> None:
    """Handle /morning command — trigger workspace session."""
    settings = get_settings()

    if message.from_user is None or message.from_user.id != settings.admin_user_id:
        await message.answer(
            "\u042d\u0442\u0430 \u043a\u043e\u043c\u0430\u043d\u0434\u0430 \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430 \u0442\u043e\u043b\u044c\u043a\u043e \u0432\u043b\u0430\u0434\u0435\u043b\u044c\u0446\u0443."
        )
        return

    if _skill is None:
        await message.answer(
            "\u0421\u0435\u0441\u0441\u0438\u044f workspace \u043d\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u0430."
        )
        return

    parsed_hours, hours_error = _parse_explicit_lookback_hours(
        message.text or "/morning"
    )
    if hours_error is not None:
        await message.answer(hours_error)
        return
    hours = parsed_hours
    if hours is None:
        hours = await _default_lookback_hours(settings)

    context = AgentContext(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        mode=mode,  # type: ignore[arg-type]
        message_id=message.message_id,
        bot=message.bot,
        metadata={"lookback_hours": hours},
    )

    if _invocation_service is None:
        await message.answer("Центральный gate навыков не настроен.")
        return

    outcome = await _invoke_morning_or_report_guard_error(
        message=message,
        context=context,
        hours=hours,
        maintenance_guard=_build_morning_maintenance_guard(settings),
    )
    if outcome is None:
        return
    if not outcome.handled or outcome.result is None:
        await message.answer("Сессия workspace не прошла через центральный gate.")
        return

    result = outcome.result

    if result.response:
        await message.answer(result.response)
