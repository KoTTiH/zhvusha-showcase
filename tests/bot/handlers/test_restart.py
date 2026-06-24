from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.handlers import restart


def _message(*, user_id: int) -> AsyncMock:
    message = AsyncMock()
    message.from_user = MagicMock()
    message.from_user.id = user_id
    message.answer = AsyncMock()
    return message


@pytest.fixture(autouse=True)
def _reset_restart_controller() -> Iterator[None]:
    restart.set_restart_controller(None)
    yield
    restart.set_restart_controller(None)


async def test_restart_admin_stops_polling_only_when_supervisor_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = MagicMock()
    controller.stop_polling = AsyncMock()
    restart.set_restart_controller(controller)

    monkeypatch.setattr(
        restart,
        "get_settings",
        lambda: SimpleNamespace(admin_user_id=12345, bot_restart_enabled=False),
    )

    stranger_message = _message(user_id=99999)
    await restart.handle_restart(stranger_message)

    stranger_message.answer.assert_awaited_once_with(
        "Эта команда доступна только владельцу."
    )
    controller.stop_polling.assert_not_awaited()

    disabled_message = _message(user_id=12345)
    await restart.handle_restart(disabled_message)

    disabled_message.answer.assert_awaited_once()
    disabled_text = disabled_message.answer.await_args.args[0]
    assert "BOT_RESTART_ENABLED" in disabled_text
    assert "supervisor" in disabled_text.lower()
    controller.stop_polling.assert_not_awaited()

    monkeypatch.setattr(
        restart,
        "get_settings",
        lambda: SimpleNamespace(admin_user_id=12345, bot_restart_enabled=True),
    )

    enabled_message = _message(user_id=12345)
    await restart.handle_restart(enabled_message)

    enabled_message.answer.assert_awaited_once()
    enabled_text = enabled_message.answer.await_args.args[0].lower()
    assert "перезапуск" in enabled_text
    controller.stop_polling.assert_awaited_once_with()
