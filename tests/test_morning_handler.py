"""Tests for bot/handlers/morning.py — /morning command handler."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.bot.handlers.morning import (
    handle_morning,
    set_invocation_service,
    set_skill,
)

_PATCH_SETTINGS = "src.bot.handlers.morning.get_settings"


def _make_message(
    user_id: int = 12345,
    text: str = "/morning",
) -> MagicMock:
    msg = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.message_id = 1
    msg.chat = MagicMock()
    msg.chat.id = 100
    msg.bot = MagicMock()
    msg.answer = AsyncMock()
    return msg


class _MorningInvocationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object, str]] = []

    async def invoke_named_skill(
        self,
        message: str,
        context: object,
        skills: object,
        skill_name: str,
    ) -> object:
        self.calls.append((message, context, skills, skill_name))
        skill = next(iter(skills))
        return SimpleNamespace(
            handled=True,
            result=await skill.execute(message, context),
        )


@pytest.mark.asyncio
async def test_morning_non_admin() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(user_id=99999)

    with patch(_PATCH_SETTINGS, return_value=settings):
        await handle_morning(msg)

    msg.answer.assert_awaited_once()
    assert "владельц" in msg.answer.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_morning_no_skill() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message()

    import src.bot.handlers.morning as mod

    old_skill = mod._skill
    old_service = mod._invocation_service
    old_skills = mod._invocation_skills
    mod._skill = None

    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        assert "не настроена" in msg.answer.call_args_list[0][0][0].lower()
    finally:
        mod._skill = old_skill
        mod._invocation_service = old_service
        mod._invocation_skills = old_skills


@pytest.mark.asyncio
async def test_morning_default_hours_uses_elapsed_since_last_consolidation(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(admin_user_id=12345, workspace_path=str(tmp_path))
    msg = _make_message(text="/morning")
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))
    service = _MorningInvocationService()
    last_consolidated = AsyncMock(return_value=10_000.0 - 7201.0)

    set_skill(mock_skill)
    set_invocation_service(service, [mock_skill])
    try:
        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(
                "src.memory.ConsolidationLock.read_last_consolidated_at",
                last_consolidated,
            ),
            patch("time.time", return_value=10_000.0),
        ):
            await handle_morning(msg)
        ctx = mock_skill.execute.call_args[0][1]
        assert ctx.metadata["lookback_hours"] == 3
        assert service.calls[0][3] == mock_skill.name
        last_consolidated.assert_awaited_once()
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_default_hours_uses_legacy_consolidation_result_mtime(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(admin_user_id=12345, workspace_path=str(tmp_path))
    msg = _make_message(text="/morning")
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))
    service = _MorningInvocationService()
    last_consolidated = AsyncMock(return_value=0.0)
    legacy_result = tmp_path / "inbox" / ".processed" / "consolidation_results.md"
    legacy_result.parent.mkdir(parents=True)
    legacy_result.write_text("Previous consolidation.", encoding="utf-8")
    now = 100_000.0
    legacy_ts = now - 9 * 3600 - 1
    os.utime(legacy_result, (legacy_ts, legacy_ts))

    set_skill(mock_skill)
    set_invocation_service(service, [mock_skill])
    try:
        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(
                "src.memory.ConsolidationLock.read_last_consolidated_at",
                last_consolidated,
            ),
            patch("time.time", return_value=now),
        ):
            await handle_morning(msg)
        ctx = mock_skill.execute.call_args[0][1]
        assert ctx.metadata["lookback_hours"] == 10
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_default_hours_uses_max_window_without_any_marker(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(admin_user_id=12345, workspace_path=str(tmp_path))
    msg = _make_message(text="/morning")
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))
    service = _MorningInvocationService()
    last_consolidated = AsyncMock(return_value=0.0)

    set_skill(mock_skill)
    set_invocation_service(service, [mock_skill])
    try:
        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(
                "src.memory.ConsolidationLock.read_last_consolidated_at",
                last_consolidated,
            ),
        ):
            await handle_morning(msg)
        ctx = mock_skill.execute.call_args[0][1]
        assert ctx.metadata["lookback_hours"] == 720
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_invocation_uses_live_skill_sequence_reference() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(text="/morning 1")
    mock_skill = AsyncMock()
    mock_skill.name = "workspace_session"
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))
    skills: list[object] = []
    service = _MorningInvocationService()

    set_skill(mock_skill)
    # main.py внедряет service до того, как _skills.extend(...) заполнит registry.
    set_invocation_service(service, skills)
    skills.append(mock_skill)
    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        assert msg.answer.call_args_list[0][0][0] == "Done"
        assert service.calls[0][2] is skills
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_custom_hours() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(text="/morning 48")
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))
    service = _MorningInvocationService()

    set_skill(mock_skill)
    set_invocation_service(service, [mock_skill])
    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        ctx = mock_skill.execute.call_args[0][1]
        assert ctx.metadata["lookback_hours"] == 48
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_hours_too_small() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(text="/morning 0")
    mock_skill = AsyncMock()
    set_skill(mock_skill)

    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        # Should reply with error, not call skill
        assert any("минимум" in c[0][0].lower() for c in msg.answer.call_args_list)
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_hours_too_large() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(text="/morning 999")
    mock_skill = AsyncMock()
    set_skill(mock_skill)

    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        assert any("максимум" in c[0][0].lower() for c in msg.answer.call_args_list)
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_hours_invalid() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message(text="/morning abc")
    mock_skill = AsyncMock()
    set_skill(mock_skill)

    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        assert any("число" in c[0][0].lower() for c in msg.answer.call_args_list)
    finally:
        set_skill(None)
        set_invocation_service(None)


@pytest.mark.asyncio
async def test_morning_no_response() -> None:
    settings = SimpleNamespace(admin_user_id=12345)
    msg = _make_message()
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value=MagicMock(response=""))

    set_skill(mock_skill)
    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(msg)
        # First answer is "Запускаю...", skill returns empty response = no second answer
        answers = [c[0][0] for c in msg.answer.call_args_list]
        assert len(answers) == 1  # only the "launching" message
    finally:
        set_skill(None)  # type: ignore[arg-type]


def test_set_skill() -> None:
    import src.bot.handlers.morning as mod

    old = mod._skill
    mock = MagicMock()
    set_skill(mock)
    assert mod._skill is mock
    mod._skill = old
