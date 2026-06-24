from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog
from aiogram import Router
from aiogram.filters import Command

from src.core.config import get_settings

if TYPE_CHECKING:
    from aiogram.types import Message

logger = structlog.get_logger()

router = Router(name="restart")


class RestartController(Protocol):
    async def stop_polling(self) -> None: ...


_controller: RestartController | None = None


def set_restart_controller(controller: RestartController | None) -> None:
    global _controller
    _controller = controller


@router.message(Command("restart"))
async def handle_restart(message: Message) -> None:
    settings = get_settings()

    if message.from_user is None or message.from_user.id != settings.admin_user_id:
        await message.answer("Эта команда доступна только владельцу.")
        return

    if not settings.bot_restart_enabled:
        await message.answer(
            "Перезапуск выключен: BOT_RESTART_ENABLED=false. "
            "Включай его только когда бот запущен под внешним supervisor "
            "вроде systemd unit `zhvusha-bot.service`; иначе я просто "
            "остановлюсь и никто не поднимет процесс заново."
        )
        return

    if _controller is None:
        await message.answer(
            "Не могу перезапуститься: polling controller не подключён."
        )
        return

    await message.answer(
        "Приняла команду на перезапуск. Останавливаю polling; дальше меня "
        "должен поднять внешний supervisor."
    )
    logger.info("bot_restart_requested", user_id=message.from_user.id)
    await _controller.stop_polling()
