"""Tests for SafetyGuard."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import structlog
from src.daemon.decision import ActionSpec, DaemonDecision, DaemonDecisionType
from src.daemon.safety import SafetyGuard, SafetyGuardConfig
from src.daemon.tools.base import DaemonTool, ToolResult
from src.daemon.tools.registry import ToolRegistry


class _SafeTool(DaemonTool):
    name = "safe_tool"
    description = "A safe tool"
    requires_approval = False

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True)


class _ApprovalTool(DaemonTool):
    name = "needs_approval_tool"
    description = "A tool requiring approval"
    requires_approval = True

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True)


def _make_registry(*tools: DaemonTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


class TestSafetyGuard:
    def test_allow_no_action(self) -> None:
        guard = SafetyGuard()
        decision = DaemonDecision(decision=DaemonDecisionType.IGNORE)
        verdict = guard.check(decision)
        assert verdict.allowed is True
        assert verdict.blocked is False

    def test_allow_safe_action(self) -> None:
        registry = _make_registry(_SafeTool())
        guard = SafetyGuard(tool_registry=registry)
        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="safe_tool", params={}),
        )
        verdict = guard.check(decision)
        assert verdict.blocked is False
        assert verdict.needs_approval is False

    def test_requires_approval_from_registry(self) -> None:
        registry = _make_registry(_ApprovalTool())
        guard = SafetyGuard(tool_registry=registry)
        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_NOTIFY,
            action=ActionSpec(tool="needs_approval_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.needs_approval is True

    def test_anti_loop_detection(self) -> None:
        config = SafetyGuardConfig(anti_loop_window=3)
        guard = SafetyGuard(config=config)

        action = ActionSpec(tool="test_tool")

        # First 3 calls OK
        for _ in range(3):
            decision = DaemonDecision(
                decision=DaemonDecisionType.ACT_SILENT, action=action
            )
            guard.check(decision)

        # 4th call should detect loop
        decision = DaemonDecision(decision=DaemonDecisionType.ACT_SILENT, action=action)
        verdict = guard.check(decision)
        assert verdict.blocked is True
        assert "цикл" in verdict.reason

    def test_requires_approval_flag_on_decision(self) -> None:
        guard = SafetyGuard()
        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_NOTIFY,
            action=ActionSpec(tool="safe_tool"),
            requires_approval=True,
        )
        verdict = guard.check(decision)
        assert verdict.needs_approval is True

    def test_unknown_tool_requires_approval(self) -> None:
        """Unknown tools require approval (fail-closed)."""
        registry = _make_registry()
        guard = SafetyGuard(tool_registry=registry)
        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="nonexistent"),
        )
        verdict = guard.check(decision)
        assert verdict.needs_approval is True

    def test_no_registry_requires_approval(self) -> None:
        """When tool_registry is None, approval is required (fail-closed)."""
        guard = SafetyGuard()  # no registry
        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="some_tool"),
        )

        with structlog.testing.capture_logs() as logs:
            verdict = guard.check(decision)

        assert verdict.needs_approval is True
        warning_logs = [
            entry for entry in logs if entry.get("event") == "safety_no_registry"
        ]
        assert len(warning_logs) == 1
        assert warning_logs[0]["tool"] == "some_tool"

    def test_anti_loop_blocks_repeated_approval_actions(self) -> None:
        """Approval-requiring tools are tracked for anti-loop detection.

        After anti_loop_window consecutive attempts with the same
        approval-requiring tool, the next attempt is blocked (not
        just sent to approval again).
        """
        approval_tool = _ApprovalTool()
        approval_tool.name = "knowledge_store"
        registry = _make_registry(approval_tool)
        config = SafetyGuardConfig(anti_loop_window=3)
        guard = SafetyGuard(config=config, tool_registry=registry)

        action = ActionSpec(tool="knowledge_store")

        # First 3 calls: needs_approval (tracked internally)
        for _ in range(3):
            decision = DaemonDecision(
                decision=DaemonDecisionType.ACT_SILENT, action=action
            )
            verdict = guard.check(decision)
            assert verdict.needs_approval is True
            assert verdict.blocked is False

        # 4th call: blocked as loop
        decision = DaemonDecision(decision=DaemonDecisionType.ACT_SILENT, action=action)
        verdict = guard.check(decision)
        assert verdict.blocked is True
        assert "цикл" in verdict.reason

    def test_tool_registry_determines_approval(self) -> None:
        approval_tool = _ApprovalTool()
        approval_tool.name = "my_tool"
        registry = _make_registry(approval_tool)
        guard = SafetyGuard(tool_registry=registry)

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_NOTIFY,
            action=ActionSpec(tool="my_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.needs_approval is True

    def test_hourly_limit_blocks(self) -> None:
        """Actions are blocked when hourly call limit is exceeded."""
        usage_tracker = MagicMock()
        usage_tracker.get_today.return_value = MagicMock(cost_usd=0.0)
        usage_tracker.get_calls_in_last_hour.return_value = 60

        config = SafetyGuardConfig(max_llm_calls_per_hour=60)
        guard = SafetyGuard(config=config, usage_tracker=usage_tracker)

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="test_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.blocked is True
        assert "вызовов/час" in verdict.reason

    def test_hourly_limit_allows_under(self) -> None:
        """Actions are allowed when under hourly limit."""
        usage_tracker = MagicMock()
        usage_tracker.get_today.return_value = MagicMock(cost_usd=0.0)
        usage_tracker.get_calls_in_last_hour.return_value = 30

        config = SafetyGuardConfig(max_llm_calls_per_hour=60)
        guard = SafetyGuard(config=config, usage_tracker=usage_tracker)

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="test_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.blocked is False

    def test_pacing_blocks_when_ahead(self) -> None:
        """Pacing blocks when spending is ahead of daily pace."""
        usage_tracker = MagicMock()
        # Simulate: $0.20 spent, but only 6h elapsed → allowed = $0.30 * 6/24 = $0.075
        usage_tracker.get_today.return_value = MagicMock(cost_usd=0.20)
        usage_tracker.get_calls_in_last_hour.return_value = 5

        config = SafetyGuardConfig(max_llm_cost_per_day_usd=0.30)
        guard = SafetyGuard(config=config, usage_tracker=usage_tracker)
        # Override hours to make test deterministic
        guard._hours_elapsed_today = lambda: 6.0  # type: ignore[method-assign]

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="test_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.blocked is True
        assert "Пейсинг" in verdict.reason

    def test_pacing_allows_when_on_track(self) -> None:
        """Pacing allows when spending is on or below pace."""
        usage_tracker = MagicMock()
        # Simulate: $0.05 spent, 12h elapsed → allowed = $0.30 * 12/24 = $0.15
        usage_tracker.get_today.return_value = MagicMock(cost_usd=0.05)
        usage_tracker.get_calls_in_last_hour.return_value = 5

        config = SafetyGuardConfig(max_llm_cost_per_day_usd=0.30)
        guard = SafetyGuard(config=config, usage_tracker=usage_tracker)
        guard._hours_elapsed_today = lambda: 12.0  # type: ignore[method-assign]

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_SILENT,
            action=ActionSpec(tool="test_tool"),
        )
        verdict = guard.check(decision)
        assert verdict.blocked is False
