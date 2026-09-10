"""Regression tests for /morning maintenance guard vs autonomous self-coding."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agent_runtime.models import ContextCapsule
from src.bot.handlers.morning import (
    handle_morning,
    set_invocation_service,
    set_skill,
)
from src.bot.main import (
    _autonomous_self_coding_loop,
    _notify_autonomous_self_coding_confirmation,
    _record_autonomous_user_activity,
)
from src.bot.maintenance_guard import (
    AutonomousSelfCodingRuntimeGuard,
    MaintenanceGuardError,
    MorningMaintenanceGuard,
    UserActivityGuard,
)
from src.skills.base import AgentContext

_PATCH_SETTINGS = "src.bot.handlers.morning.get_settings"
_PATCH_MAIN_SETTINGS = "src.bot.main.get_settings"


class _Skill:
    def __init__(self) -> None:
        self.calls = 0

    async def run_once(self) -> object:
        self.calls += 1
        return SimpleNamespace(success=True, metadata={"calls": self.calls})


def test_morning_maintenance_guard_ignores_stale_marker(tmp_path: Path) -> None:
    marker = MorningMaintenanceGuard(
        marker_path=tmp_path / "runtime" / "maintenance" / "morning.json",
        ttl_seconds=60,
    )
    marker.mark_active(owner="test", token="test-token", now=1000.0)

    assert marker.is_active(now=1059.0) is True
    assert marker.is_active(now=1061.0) is False


def test_morning_maintenance_guard_preserves_overlapping_windows(
    tmp_path: Path,
) -> None:
    marker = MorningMaintenanceGuard(
        marker_path=tmp_path / "runtime" / "maintenance" / "morning.json",
        ttl_seconds=60,
    )

    with marker.active(owner="first"):
        assert marker.is_active() is True
        with marker.active(owner="second"):
            assert marker.is_active() is True
        assert marker.is_active() is True
    assert marker.is_active() is False


def test_morning_maintenance_guard_fails_closed_when_marker_write_fails(
    tmp_path: Path,
) -> None:
    marker = MorningMaintenanceGuard(
        marker_path=tmp_path / "runtime" / "maintenance" / "morning.json",
        ttl_seconds=60,
    )

    with (
        patch("src.bot.maintenance_guard._write_json", return_value=False),
        pytest.raises(MaintenanceGuardError),
        marker.active(owner="test"),
    ):
        pass


def test_autonomous_guard_fails_closed_on_corrupt_state(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "autonomous_self_coding.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{not-json", encoding="utf-8")
    runtime_guard = AutonomousSelfCodingRuntimeGuard(
        state_path=state_path,
        morning_guard=None,
        restart_throttle_seconds=3600,
    )

    decision = runtime_guard.should_run()

    assert decision.should_run is False
    assert decision.reason == "persistent_state_unreadable"


def test_autonomous_guard_waits_until_user_is_idle(tmp_path: Path) -> None:
    activity = UserActivityGuard(
        activity_path=tmp_path / "runtime" / "user_activity.json",
        idle_seconds=7200,
    )
    activity.record_activity(source="telegram", now=1000.0)
    runtime_guard = AutonomousSelfCodingRuntimeGuard(
        state_path=tmp_path / "runtime" / "autonomous_self_coding.json",
        morning_guard=None,
        restart_throttle_seconds=0,
        user_activity_guard=activity,
    )

    recent = runtime_guard.should_run(now=1000.0 + 7199)
    idle = runtime_guard.should_run(now=1000.0 + 7200)

    assert recent.should_run is False
    assert recent.reason == "user_recently_active"
    assert recent.next_allowed_at == 8200.0
    assert idle.should_run is True


def test_autonomous_guard_fails_closed_when_user_activity_is_unknown(
    tmp_path: Path,
) -> None:
    runtime_guard = AutonomousSelfCodingRuntimeGuard(
        state_path=tmp_path / "runtime" / "autonomous_self_coding.json",
        morning_guard=None,
        restart_throttle_seconds=0,
        user_activity_guard=UserActivityGuard(
            activity_path=tmp_path / "runtime" / "user_activity.json",
            idle_seconds=7200,
        ),
    )

    decision = runtime_guard.should_run(now=1000.0)

    assert decision.should_run is False
    assert decision.reason == "user_activity_unknown"


def test_incoming_admin_message_records_autonomous_user_activity(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        admin_user_id=12345,
        workspace_path=str(tmp_path),
        autonomous_self_coding_user_idle_seconds=7200,
        autonomous_self_coding_user_activity_path="",
    )

    with patch(_PATCH_MAIN_SETTINGS, return_value=settings):
        _record_autonomous_user_activity(
            AgentContext(
                user_id=12345,
                chat_id=12345,
                mode="personal",
                message_id=99,
                metadata={"source": "telegram", "interface": "telegram"},
            )
        )

    payload = json.loads(
        (tmp_path / "runtime" / "user_activity.json").read_text(encoding="utf-8")
    )
    assert payload["source"] == "telegram"
    assert payload["metadata"]["message_id"] == "99"


@pytest.mark.asyncio
async def test_autonomous_loop_skips_when_started_state_cannot_be_persisted(
    tmp_path: Path,
) -> None:
    runtime_guard = AutonomousSelfCodingRuntimeGuard(
        state_path=tmp_path / "runtime" / "autonomous_self_coding.json",
        morning_guard=None,
        restart_throttle_seconds=3600,
    )
    skill = _Skill()

    with patch("src.bot.maintenance_guard._write_json", return_value=False):
        await _run_one_loop_turn(skill, runtime_guard)

    assert skill.calls == 0


@pytest.mark.asyncio
async def test_autonomous_confirmation_notification_asks_admin() -> None:
    class BotStub:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_message(
            self,
            *,
            chat_id: int | str,
            text: str,
            parse_mode: str | None = None,
        ) -> object:
            self.messages.append(
                {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
            )
            return object()

    bot = BotStub()
    completed = SimpleNamespace(
        result=ContextCapsule(
            summary="Нужен свежий апрув.",
            artifacts=(
                "old-spec",
                "spec_slug: old-spec",
                "needs_user_confirmation:true",
            ),
        )
    )

    await _notify_autonomous_self_coding_confirmation(
        bot=bot,  # type: ignore[arg-type]
        admin_user_id=123,
        completed=completed,
    )

    assert bot.messages == [
        {
            "chat_id": 123,
            "text": (
                "Нашла `old-spec`, но сама не запускаю.\n\n"
                "Нужно свежее решение: это ещё надо делать или уже нет?"
                "\n\nЕсли да — напиши: `запусти spec old-spec`."
            ),
            "parse_mode": None,
        }
    ]


async def _run_one_loop_turn(
    skill: _Skill,
    guard: AutonomousSelfCodingRuntimeGuard,
) -> None:
    async def _cancel_on_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    with (
        patch("src.bot.main.asyncio.sleep", _cancel_on_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await _autonomous_self_coding_loop(
            skill=skill,  # type: ignore[arg-type]
            interval_seconds=3600,
            initial_delay_seconds=0,
            runtime_guard=guard,
        )


@pytest.mark.asyncio
async def test_autonomous_self_coding_waits_for_persistent_interval_and_skips_active_morning(
    tmp_path: Path,
) -> None:
    """Regression anchored to archive node morning-recovery-window-archive-db-failed-177965234."""
    state_path = tmp_path / "runtime" / "autonomous_self_coding.json"
    marker_path = tmp_path / "runtime" / "maintenance" / "morning.json"
    runtime_guard = AutonomousSelfCodingRuntimeGuard(
        state_path=state_path,
        morning_guard=MorningMaintenanceGuard(
            marker_path=marker_path,
            ttl_seconds=3600,
        ),
        restart_throttle_seconds=3600,
    )
    skill = _Skill()

    runtime_guard.record_run(status="complete")
    await _run_one_loop_turn(skill, runtime_guard)
    assert skill.calls == 0

    state_path.write_text(
        json.dumps({"last_run_at": 1.0, "status": "complete"}),
        encoding="utf-8",
    )
    with MorningMaintenanceGuard(marker_path=marker_path, ttl_seconds=3600).active(
        owner="test_morning"
    ):
        await _run_one_loop_turn(skill, runtime_guard)
    assert skill.calls == 0

    await _run_one_loop_turn(skill, runtime_guard)
    assert skill.calls == 1
    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted_state["last_run_at"] > 1.0
    assert persisted_state["status"] == "complete"


class _MorningInvocationService:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path
        self.marker_was_active = False

    async def invoke_named_skill(
        self,
        message: str,
        context: object,
        skills: object,
        skill_name: str,
    ) -> object:
        self.marker_was_active = MorningMaintenanceGuard(
            marker_path=self.marker_path,
            ttl_seconds=3600,
        ).is_active()
        skill = next(iter(skills))
        return SimpleNamespace(
            handled=True,
            result=await skill.execute(message, context),
        )


def _make_message() -> MagicMock:
    msg = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = 12345
    msg.text = "/morning 1"
    msg.message_id = 1
    msg.chat = MagicMock()
    msg.chat.id = 100
    msg.bot = MagicMock()
    msg.answer = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_morning_handler_marks_maintenance_window_during_invocation(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        admin_user_id=12345,
        workspace_path=str(tmp_path),
        autonomous_self_coding_morning_guard_enabled=True,
    )
    marker_path = tmp_path / "runtime" / "maintenance" / "morning.json"
    service = _MorningInvocationService(marker_path)
    mock_skill = AsyncMock()
    mock_skill.name = "workspace_session"
    mock_skill.execute = AsyncMock(return_value=MagicMock(response="Done"))

    set_skill(mock_skill)
    set_invocation_service(service, [mock_skill])
    try:
        with patch(_PATCH_SETTINGS, return_value=settings):
            await handle_morning(_make_message())
    finally:
        set_skill(None)
        set_invocation_service(None)

    assert service.marker_was_active is True
    assert not marker_path.exists()
