"""Tests for daemon/main.py — ZhvushaDaemon orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.daemon.decision import DaemonDecisionType
from src.daemon.main import ZhvushaDaemon
from src.daemon.tools.base import ToolResult


def _make_daemon(
    *,
    sleep_agent: AsyncMock | None = None,
    life_runtime_runner: MagicMock | None = None,
    life_runtime_enabled: bool = False,
    admin_chat_id: int = 12345,
) -> tuple[ZhvushaDaemon, dict[str, AsyncMock]]:
    mocks = {
        "stream": AsyncMock(),
        "decision": AsyncMock(),
        "safety": MagicMock(),
        "tools": AsyncMock(),
        "audit": AsyncMock(),
        "approval": AsyncMock(),
    }
    mocks["stream"].ensure_groups = AsyncMock()
    mocks["stream"].read_priority = AsyncMock(return_value=[])
    mocks["stream"].ack = AsyncMock()
    mocks["stream"].start_wake_listener = AsyncMock()
    mocks["audit"].record = AsyncMock(return_value=1)
    mocks["approval"].get_approved = AsyncMock(return_value=[])
    mocks["approval"].recover_stuck = AsyncMock()

    daemon = ZhvushaDaemon(
        signal_stream=mocks["stream"],
        decision_engine=mocks["decision"],
        safety_guard=mocks["safety"],
        tool_registry=mocks["tools"],
        audit_log=mocks["audit"],
        approval_store=mocks["approval"],
        sleep_agent=sleep_agent,
        life_runtime_runner=life_runtime_runner,
        life_runtime_enabled=life_runtime_enabled,
        admin_chat_id=admin_chat_id,
    )
    return daemon, mocks


def _signal(
    signal_id: str = "sig-1",
    source: str = "test",
    signal_type: str = "test_event",
) -> MagicMock:
    sig = MagicMock()
    sig.id = signal_id
    sig.source = source
    sig.signal_type = signal_type
    sig.stream_entry_id = b"1-0"
    sig.priority = "normal"
    return sig


def _decision(
    dtype: DaemonDecisionType = DaemonDecisionType.ACT_SILENT,
    tool: str | None = "send_telegram",
    params: dict | None = None,
) -> MagicMock:
    d = MagicMock()
    d.decision = dtype
    d.reasoning = "test reason"
    if tool:
        d.action = MagicMock()
        d.action.tool = tool
        d.action.params = params or {"text": "hi"}
    else:
        d.action = None
    return d


# --- stop ---


@pytest.mark.asyncio
async def test_stop_sets_running_false() -> None:
    daemon, _ = _make_daemon()
    daemon._running = True
    await daemon.stop()
    assert daemon._running is False


# --- _process_signal: blocked ---


@pytest.mark.asyncio
async def test_process_signal_blocked() -> None:
    daemon, mocks = _make_daemon()
    sig = _signal()
    dec = _decision()
    mocks["decision"].decide = AsyncMock(return_value=dec)
    mocks["safety"].check.return_value = MagicMock(
        blocked=True, needs_approval=False, reason="anti-loop"
    )

    await daemon._process_signal(sig)

    mocks["audit"].record.assert_awaited_once()
    call_kwargs = mocks["audit"].record.call_args.kwargs
    assert call_kwargs["result"] == "blocked"
    mocks["stream"].ack.assert_awaited_once()


# --- _process_signal: ignore ---


@pytest.mark.asyncio
async def test_process_signal_ignore() -> None:
    daemon, mocks = _make_daemon()
    sig = _signal()
    dec = _decision(DaemonDecisionType.IGNORE, tool=None)
    mocks["decision"].decide = AsyncMock(return_value=dec)
    mocks["safety"].check.return_value = MagicMock(
        blocked=False, needs_approval=False, reason=""
    )

    await daemon._process_signal(sig)

    call_kwargs = mocks["audit"].record.call_args.kwargs
    assert call_kwargs["decision"] == "ignore"
    mocks["stream"].ack.assert_awaited_once()


# --- _process_signal: execute action ---


@pytest.mark.asyncio
async def test_process_signal_execute() -> None:
    daemon, mocks = _make_daemon()
    sig = _signal()
    dec = _decision(DaemonDecisionType.ACT_SILENT, tool="send_telegram")
    mocks["decision"].decide = AsyncMock(return_value=dec)
    mocks["safety"].check.return_value = MagicMock(
        blocked=False, needs_approval=False, reason=""
    )
    mocks["tools"].execute = AsyncMock(
        return_value=ToolResult(success=True, message="sent")
    )

    await daemon._process_signal(sig)

    mocks["tools"].execute.assert_awaited_once()
    call_kwargs = mocks["audit"].record.call_args.kwargs
    assert call_kwargs["result"] == "success"


# --- _process_signal: needs approval ---


@pytest.mark.asyncio
async def test_process_signal_needs_approval() -> None:
    daemon, mocks = _make_daemon()
    sig = _signal()
    dec = _decision(DaemonDecisionType.ACT_NOTIFY)
    mocks["decision"].decide = AsyncMock(return_value=dec)
    mocks["safety"].check.return_value = MagicMock(
        blocked=False, needs_approval=True, reason="requires_approval"
    )
    mocks["approval"].create = AsyncMock(return_value=1)
    # Mock send_telegram tool
    notify_tool = AsyncMock()
    notify_tool.execute = AsyncMock(
        return_value=ToolResult(success=True, message="ok", data={"message_id": 42})
    )
    mocks["tools"].get = MagicMock(return_value=notify_tool)

    await daemon._process_signal(sig)

    mocks["approval"].create.assert_awaited_once()
    mocks["approval"].set_telegram_message_id.assert_awaited_once_with(1, 42)
    call_kwargs = mocks["audit"].record.call_args.kwargs
    assert call_kwargs["result"] == "pending_approval"


@pytest.mark.asyncio
async def test_process_signal_needs_approval_no_action() -> None:
    daemon, mocks = _make_daemon()
    sig = _signal()
    dec = _decision(DaemonDecisionType.ACT_NOTIFY, tool=None)
    mocks["decision"].decide = AsyncMock(return_value=dec)
    mocks["safety"].check.return_value = MagicMock(
        blocked=False, needs_approval=True, reason="requires_approval"
    )

    await daemon._process_signal(sig)

    call_kwargs = mocks["audit"].record.call_args.kwargs
    assert call_kwargs["result"] == "skipped"


# --- _execute_approved_actions ---


@pytest.mark.asyncio
async def test_execute_approved_action() -> None:
    daemon, mocks = _make_daemon()
    action = MagicMock()
    action.id = 1
    action.signal_id = "sig-1"
    action.tool_name = "send_telegram"
    action.tool_params = {"text": "ok"}
    action.decision_type = "act_approval"
    mocks["approval"].get_approved = AsyncMock(return_value=[action])
    mocks["approval"].mark_executing = AsyncMock(return_value=True)
    mocks["tools"].execute = AsyncMock(
        return_value=ToolResult(success=True, message="done")
    )

    await daemon._execute_approved_actions()

    mocks["approval"].mark_executing.assert_awaited_once_with(1)
    mocks["tools"].execute.assert_awaited_once()
    mocks["approval"].mark_executed.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_approved_claim_fails() -> None:
    daemon, mocks = _make_daemon()
    action = MagicMock()
    action.id = 1
    mocks["approval"].get_approved = AsyncMock(return_value=[action])
    mocks["approval"].mark_executing = AsyncMock(return_value=False)

    await daemon._execute_approved_actions()

    mocks["tools"].execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_approved_tool_error() -> None:
    daemon, mocks = _make_daemon()
    action = MagicMock()
    action.id = 1
    action.signal_id = "sig-1"
    action.tool_name = "bad_tool"
    action.tool_params = {}
    action.decision_type = "act_approval"
    mocks["approval"].get_approved = AsyncMock(return_value=[action])
    mocks["approval"].mark_executing = AsyncMock(return_value=True)
    mocks["tools"].execute = AsyncMock(side_effect=RuntimeError("crash"))

    await daemon._execute_approved_actions()

    # Should mark as failed
    mocks["approval"].mark_executed.assert_awaited_once_with(1, success=False)


@pytest.mark.asyncio
async def test_execute_approved_recover_stuck() -> None:
    daemon, mocks = _make_daemon()
    daemon._last_recover = 0.0
    daemon._RECOVER_INTERVAL = 0  # always recover

    await daemon._execute_approved_actions()

    mocks["approval"].recover_stuck.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_idle_life_runtime_hook_is_disabled_by_default() -> None:
    sleep_agent = AsyncMock()
    sleep_agent.run_maintenance_cycle = AsyncMock(return_value=0)
    life_runtime_runner = MagicMock()
    daemon, _ = _make_daemon(
        sleep_agent=sleep_agent,
        life_runtime_runner=life_runtime_runner,
    )

    await daemon._run_idle_maintenance(woken=False, signal_count=0)

    life_runtime_runner.run_once.assert_not_called()
    sleep_agent.run_maintenance_cycle.assert_awaited_once()
