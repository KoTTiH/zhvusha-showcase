"""Integration tests for KworkMonitorSkill.

Ported from tests/test_kwork_skill.py in phase 7.2. Command handling tests
moved to ``test_handlers.py`` because ``/kwork /sleep /wake`` are now
routed via aiogram ``Router`` rather than the BaseSkill dispatch path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from src.skills.kwork_monitor.skill import KworkMonitorSkill
from structlog.testing import capture_logs


def _settings(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "kwork_login": "test_login",
        "kwork_password": "test_pass",
        "kwork_phone_last": "",
        "kwork_keywords": "python,telegram",
        "kwork_min_budget": 3000,
        "kwork_max_offers": 15,
        "kwork_poll_interval_seconds": 60,
        "admin_user_id": 12345,
        "redis_url": "redis://localhost:6379/0",
        "google_api_key": "test-key",
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _proj(
    id: int = 1,
    title: str = "Python бот",
    description: str = "Нужен Telegram бот",
    price: int = 5000,
    offers: int = 3,
    username: str = "client",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        title=title,
        description=description,
        price=price,
        offers=offers,
        username=username,
    )


@pytest.fixture
def skill() -> KworkMonitorSkill:
    return KworkMonitorSkill()


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.sismember = AsyncMock(return_value=False)
    redis.smembers = AsyncMock(return_value=set())
    redis.sadd = AsyncMock()
    redis.expire = AsyncMock()
    redis.aclose = AsyncMock()
    return redis


class TestPollOnce:
    async def test_new_monitor_starts_sleeping_by_default(
        self, skill: KworkMonitorSkill, mock_redis: AsyncMock
    ) -> None:
        skill._redis = mock_redis
        mock_client = AsyncMock()
        mock_client.get_projects = AsyncMock(return_value=[_proj(id=1)])
        skill._kwork_client = mock_client
        bot = AsyncMock()

        await skill._poll_once(bot)

        bot.send_message.assert_not_awaited()
        assert skill.is_sleeping is True
        assert skill._poll_count == 0
        mock_redis.smembers.assert_not_awaited()
        mock_client.get_projects.assert_not_awaited()

    async def test_poll_loop_logs_sanitized_network_errors(
        self, skill: KworkMonitorSkill
    ) -> None:
        secret_error = (
            "Kwork body login='private_login' password='super_secret' "
            "phone_last='6755' token='abc123' Authorization: Basic abcdef"
        )
        skill._poll_once = AsyncMock(side_effect=RuntimeError(secret_error))  # type: ignore[method-assign]

        with (
            patch(
                "src.skills.kwork_monitor.skill.get_settings", return_value=_settings()
            ),
            patch(
                "src.skills.kwork_monitor.skill.asyncio.sleep",
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
            capture_logs() as logs,
            pytest.raises(asyncio.CancelledError),
        ):
            await skill._poll_loop(AsyncMock())

        [event] = [log for log in logs if log["event"] == "kwork_poll_error"]
        rendered = repr(event)

        assert event["error_type"] == "RuntimeError"
        assert "exc_info" not in event
        assert "exception" not in event
        assert "traceback" not in rendered.lower()
        assert "super_secret" not in rendered
        assert "private_login" not in rendered
        assert "6755" not in rendered
        assert "abc123" not in rendered
        assert "abcdef" not in rendered

    async def test_sends_matching_projects(
        self, skill: KworkMonitorSkill, mock_redis: AsyncMock
    ) -> None:
        await skill.wake()
        skill._redis = mock_redis
        bot = AsyncMock()
        raw = [_proj(id=1), _proj(id=2, title="Дизайн", description="Логотип")]

        mock_client = AsyncMock()
        mock_client.get_projects = AsyncMock(return_value=raw)
        skill._kwork_client = mock_client

        with patch(
            "src.skills.kwork_monitor.skill.get_settings", return_value=_settings()
        ):
            await skill._poll_once(bot)

        bot.send_message.assert_awaited_once()
        call_kwargs = bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["parse_mode"] == "HTML"
        assert skill._poll_count == 1

    async def test_skips_seen_projects(
        self, skill: KworkMonitorSkill, mock_redis: AsyncMock
    ) -> None:
        await skill.wake()
        skill._redis = mock_redis
        mock_redis.smembers = AsyncMock(return_value={b"1"})
        bot = AsyncMock()
        raw = [_proj(id=1)]

        mock_client = AsyncMock()
        mock_client.get_projects = AsyncMock(return_value=raw)
        skill._kwork_client = mock_client

        with patch(
            "src.skills.kwork_monitor.skill.get_settings", return_value=_settings()
        ):
            await skill._poll_once(bot)

        bot.send_message.assert_not_awaited()

    async def test_marks_seen(
        self, skill: KworkMonitorSkill, mock_redis: AsyncMock
    ) -> None:
        await skill.wake()
        skill._redis = mock_redis
        bot = AsyncMock()
        raw = [_proj(id=42)]

        mock_client = AsyncMock()
        mock_client.get_projects = AsyncMock(return_value=raw)
        skill._kwork_client = mock_client

        with patch(
            "src.skills.kwork_monitor.skill.get_settings", return_value=_settings()
        ):
            await skill._poll_once(bot)

        mock_redis.sadd.assert_awaited_with("kwork:seen", "42")

    async def test_skipped_when_sleeping(
        self, skill: KworkMonitorSkill, mock_redis: AsyncMock
    ) -> None:
        skill._redis = mock_redis
        bot = AsyncMock()

        await skill.sleep(8)
        await skill._poll_once(bot)

        bot.send_message.assert_not_awaited()
        assert skill._poll_count == 0


class TestSleepWake:
    async def test_sleep_pauses_monitoring(self, skill: KworkMonitorSkill) -> None:
        response = await skill.sleep()
        assert "приостановлен" in response
        assert "8ч" in response
        assert skill.is_sleeping is True

    async def test_sleep_custom_hours(self, skill: KworkMonitorSkill) -> None:
        response = await skill.sleep(6)
        assert "6ч" in response
        assert skill.is_sleeping is True

    async def test_wake_resumes(self, skill: KworkMonitorSkill) -> None:
        await skill.sleep(8)
        assert skill.is_sleeping is True
        response = await skill.wake()
        assert "возобновлён" in response
        assert skill.is_sleeping is False

    async def test_wake_when_not_sleeping(self, skill: KworkMonitorSkill) -> None:
        await skill.wake()
        response = await skill.wake()
        assert "и так активен" in response


class TestStatusCommand:
    async def test_format_status_running(self, skill: KworkMonitorSkill) -> None:
        await skill.wake()
        status = skill.format_status()
        assert "остановлен" in status
        assert "0" in status

    async def test_format_status_sleeping(self, skill: KworkMonitorSkill) -> None:
        await skill.sleep(8)
        status = skill.format_status()
        assert "спит" in status

    async def test_handle_status_command_first_call(
        self, skill: KworkMonitorSkill
    ) -> None:
        """First call — no previous messages to delete, sends status."""
        bot = AsyncMock()
        sent = AsyncMock()
        sent.message_id = 200
        bot.send_message = AsyncMock(return_value=sent)

        await skill.handle_status_command(bot=bot, chat_id=100, command_message_id=50)

        bot.delete_message.assert_not_awaited()
        bot.send_message.assert_awaited_once()
        assert skill._last_command_msg_id == 50
        assert skill._last_status_msg_id == 200

    async def test_handle_status_command_deletes_old_pair(
        self, skill: KworkMonitorSkill
    ) -> None:
        """Second call — deletes both old command and old status."""
        bot = AsyncMock()
        sent = AsyncMock()
        sent.message_id = 300
        bot.send_message = AsyncMock(return_value=sent)

        skill._last_command_msg_id = 50
        skill._last_status_msg_id = 200
        skill._status_chat_id = 100

        await skill.handle_status_command(bot=bot, chat_id=100, command_message_id=60)

        bot.delete_message.assert_any_await(chat_id=100, message_id=50)
        bot.delete_message.assert_any_await(chat_id=100, message_id=200)
        assert bot.delete_message.await_count == 2
        assert skill._last_command_msg_id == 60
        assert skill._last_status_msg_id == 300
