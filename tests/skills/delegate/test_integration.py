"""Integration tests for DelegateSkill — Codex delegation.

Ported from tests/test_delegate_skill.py in phase 7.1. Contexts use the v4
``AgentContext`` frozen dataclass instead of the legacy Pydantic model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.skills.base import AgentContext
from src.skills.delegate.skill import DelegateSkill, _backend_available

_PATCH_SETTINGS = "src.skills.delegate.skill.get_settings"


def _settings(delegate_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        delegate_enabled=delegate_enabled,
        delegate_cwd="~/Projects/ZHVUSHA",
        delegate_timeout_seconds=60,
        delegate_max_concurrent=1,
        delegate_model="test-model",
        code_agent_model="",
        codex_cli_path="codex",
        admin_user_id=12345,
    )


def _context(user_id: int = 12345, mode: str = "personal") -> AgentContext:
    return AgentContext(
        user_id=user_id,
        chat_id=user_id,
        mode=mode,  # type: ignore[arg-type]
        message_id=1,
        bot=AsyncMock(),
    )


class TestCanHandle:
    async def test_handles_delegate_command(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings()):
            score = await skill.can_handle("/delegate do something", _context())
        assert score == 1.0

    async def test_ignores_non_delegate(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings()):
            score = await skill.can_handle("hello", _context())
        assert score == 0.0

    async def test_ignores_when_disabled(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings(delegate_enabled=False)):
            score = await skill.can_handle("/delegate test", _context())
        assert score == 0.0

    async def test_ignores_non_admin(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings()):
            score = await skill.can_handle("/delegate test", _context(user_id=99999))
        assert score == 0.0


class TestExecute:
    async def test_empty_task_returns_help(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings()):
            result = await skill.execute("/delegate", _context())
        assert not result.success
        assert "Укажи задачу" in result.response

    async def test_backend_error_returns_message(self) -> None:
        skill = DelegateSkill()
        with (
            patch(_PATCH_SETTINGS, return_value=_settings()),
            patch(
                "src.skills.delegate.skill._run_delegate",
                side_effect=FileNotFoundError("missing"),
            ),
        ):
            result = await skill.execute("/delegate do something", _context())
        assert not result.success
        assert "ошибка" in result.response.lower()

    async def test_successful_delegation(self) -> None:
        skill = DelegateSkill()
        with (
            patch(_PATCH_SETTINGS, return_value=_settings()),
            patch(
                "src.skills.delegate.skill._run_delegate",
                return_value="Задача выполнена успешно!",
            ),
        ):
            result = await skill.execute("/delegate найди баги в логах", _context())
        assert result.success
        assert "Задача выполнена" in result.response

    async def test_timeout_returns_error(self) -> None:
        import asyncio

        skill = DelegateSkill()
        with (
            patch(_PATCH_SETTINGS, return_value=_settings()),
            patch(
                "src.skills.delegate.skill._run_delegate",
                side_effect=asyncio.TimeoutError,
            ),
        ):
            result = await skill.execute("/delegate long task", _context())
        assert not result.success
        assert "таймаут" in result.response.lower()

    async def test_exception_returns_error(self) -> None:
        skill = DelegateSkill()
        with (
            patch(_PATCH_SETTINGS, return_value=_settings()),
            patch(
                "src.skills.delegate.skill._run_delegate",
                side_effect=RuntimeError("crash"),
            ),
        ):
            result = await skill.execute("/delegate broken task", _context())
        assert not result.success
        assert "ошибка" in result.response.lower()

    async def test_truncation_on_long_result(self) -> None:
        skill = DelegateSkill()
        with (
            patch(_PATCH_SETTINGS, return_value=_settings()),
            patch(
                "src.skills.delegate.skill._run_delegate",
                return_value="x" * 5000,
            ),
        ):
            result = await skill.execute("/delegate big output", _context())
        assert result.success
        assert "обрезано" in result.response


class TestPrepare:
    async def test_prepare_returns_delegated_plan(self) -> None:
        skill = DelegateSkill()
        with patch(_PATCH_SETTINGS, return_value=_settings()):
            plan = await skill.prepare("/delegate do something", _context())
        assert plan.skill_name == "delegate"
        assert plan.skill_type == "delegated"
        assert plan.delegated_to == "codex_cli"
        assert "do something" in plan.human_summary


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        from src.skills.delegate.skill import _truncate

        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self) -> None:
        from src.skills.delegate.skill import _truncate

        result = _truncate("x" * 200, 100)
        assert len(result) <= 120
        assert "обрезано" in result


class TestSafeEdit:
    async def test_edit_success(self) -> None:
        from src.skills.delegate.skill import _safe_edit

        bot = AsyncMock()
        await _safe_edit(bot, chat_id=1, message_id=1, text="hi")
        bot.edit_message_text.assert_awaited_once()

    async def test_edit_error_swallowed(self) -> None:
        from src.skills.delegate.skill import _safe_edit

        bot = AsyncMock()
        bot.edit_message_text = AsyncMock(side_effect=Exception("rate limit"))
        await _safe_edit(bot, chat_id=1, message_id=1, text="hi")


class TestBackendAvailable:
    def test_returns_bool(self) -> None:
        result = _backend_available()
        assert isinstance(result, bool)
