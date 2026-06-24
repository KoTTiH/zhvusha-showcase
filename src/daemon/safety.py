"""Safety guard for the daemon — budget pacing, anti-loop, approval gate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.daemon.decision import DaemonDecision
    from src.daemon.tools.registry import ToolRegistry
    from src.monitoring.usage_tracker import UsageTracker

logger = structlog.get_logger()


@dataclass
class SafetyVerdict:
    """Result of safety check."""

    allowed: bool = True
    blocked: bool = False
    needs_approval: bool = False
    reason: str = ""


@dataclass
class SafetyGuardConfig:
    """Configuration for safety limits."""

    max_llm_cost_per_day_usd: float = 5.0
    max_llm_calls_per_hour: int = 60
    anti_loop_window: int = 3


class SafetyGuard:
    """Three-layer safety for daemon actions."""

    def __init__(
        self,
        config: SafetyGuardConfig | None = None,
        usage_tracker: UsageTracker | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._config = config or SafetyGuardConfig()
        self._usage_tracker = usage_tracker
        self._tool_registry = tool_registry
        self._recent_actions: deque[str] = deque(maxlen=self._config.anti_loop_window)

    def check(self, decision: DaemonDecision) -> SafetyVerdict:
        """Check if a decision is safe to execute."""
        if decision.action is None:
            return SafetyVerdict()

        tool_name = decision.action.tool

        # Check 1: Budget pacing — spread budget evenly across 24h
        if self._usage_tracker is not None and self._is_ahead_of_pace():
            return SafetyVerdict(
                blocked=True,
                reason=self._pace_reason(),
            )

        # Check 1b: Hourly rate limit
        if self._usage_tracker is not None and self._is_over_hourly_limit():
            return SafetyVerdict(
                blocked=True,
                reason=(
                    f"Лимит {self._config.max_llm_calls_per_hour} вызовов/час превышен"
                ),
            )

        # Check 2: Anti-loop
        if self._is_loop(tool_name):
            return SafetyVerdict(
                blocked=True,
                reason=f"Обнаружен цикл: {tool_name} вызван {self._config.anti_loop_window} раз подряд",
            )

        # Track action for anti-loop (before approval gate — loops of
        # approval-requiring actions must be detected too)
        self._recent_actions.append(tool_name)

        # Check 3: Approval gate
        if self._requires_approval(tool_name) or decision.requires_approval:
            return SafetyVerdict(
                needs_approval=True,
                reason=f"Действие {tool_name} требует подтверждения Никиты",
            )

        return SafetyVerdict()

    def _hours_elapsed_today(self) -> float:
        """Hours elapsed since midnight UTC."""
        now = datetime.now(tz=UTC)
        return now.hour + now.minute / 60.0 + now.second / 3600.0

    def _is_ahead_of_pace(self) -> bool:
        """Check if spending is ahead of the daily pace.

        Allowed budget at any point = daily_budget * (hours_elapsed / 24).
        This spreads the budget evenly: $0.30 → ~$0.0125/hr, $1 → ~$0.042/hr.
        """
        if self._usage_tracker is None:
            return False
        try:
            stats = self._usage_tracker.get_today()
            hours = max(self._hours_elapsed_today(), 0.5)  # floor to avoid /0
            allowed = self._config.max_llm_cost_per_day_usd * (hours / 24.0)
            return stats.cost_usd >= allowed
        except Exception:
            return False

    def _pace_reason(self) -> str:
        """Human-readable pacing block reason."""
        try:
            stats = self._usage_tracker.get_today()  # type: ignore[union-attr]
            hours = max(self._hours_elapsed_today(), 0.5)
            allowed = self._config.max_llm_cost_per_day_usd * (hours / 24.0)
            return (
                f"Пейсинг: потрачено ${stats.cost_usd:.3f}, "
                f"допустимо ${allowed:.3f} к {hours:.1f}ч "
                f"(бюджет ${self._config.max_llm_cost_per_day_usd}/день)"
            )
        except Exception:
            return "Пейсинг: бюджет превышен"

    def _is_over_hourly_limit(self) -> bool:
        """Check if hourly LLM call limit is exceeded."""
        if self._usage_tracker is None:
            return False
        try:
            calls = self._usage_tracker.get_calls_in_last_hour()
            return calls >= self._config.max_llm_calls_per_hour
        except Exception:
            return False

    def _is_loop(self, tool_name: str) -> bool:
        """Check if the same action has been repeated too many times."""
        if len(self._recent_actions) < self._config.anti_loop_window:
            return False
        return all(a == tool_name for a in self._recent_actions)

    def _requires_approval(self, tool_name: str) -> bool:
        """Check if a tool always requires human approval via registry.

        Fail-closed: if the registry is missing, approval is required.
        """
        if self._tool_registry is None:
            logger.warning(
                "safety_no_registry",
                tool=tool_name,
                msg="tool_registry не задан — требуется approval по умолчанию",
            )
            return True
        tool = self._tool_registry.get(tool_name)
        if tool is None:
            return True
        return tool.requires_approval
