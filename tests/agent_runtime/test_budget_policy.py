"""Autonomous model/effort/budget policy contract tests."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

REQUIRED_ENV = {
    "BOT_TOKEN": "test_token",
    "CHANNEL_ID": "@test_channel",
    "ADMIN_USER_ID": "12345",
}


def _settings(**extra: str):
    from src.core.config import Settings

    with patch.dict(os.environ, {**REQUIRED_ENV, **extra}, clear=True):
        return Settings(_env_file=None)  # type: ignore[call-arg]


def test_budget_policy_routes_cheap_autonomous_jobs_to_worker_low_effort() -> None:
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetUsageSnapshot,
    )

    policy = AgentBudgetPolicy.from_settings(_settings())

    decision = policy.evaluate(
        BudgetJobKind.SUMMARIZATION,
        BudgetUsageSnapshot(estimated_cost_usd=Decimal("0.01")),
    )

    assert decision.decision is BudgetDecisionType.ALLOW
    assert decision.route.tier == "worker"
    assert decision.route.provider == "codex_cli"
    assert decision.route.model == "default"
    assert decision.route.reasoning_effort == "low"
    assert decision.route.auto_run_allowed is True
    assert any("дневной лимит" in line for line in decision.status_lines)


def test_budget_policy_blocks_when_daily_or_weekly_cap_is_exceeded() -> None:
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetUsageSnapshot,
    )

    policy = AgentBudgetPolicy.from_settings(
        _settings(
            AUTONOMOUS_LOOP_BUDGET_DAILY_USD="0.10",
            AUTONOMOUS_LOOP_BUDGET_WEEKLY_USD="0.20",
        )
    )

    daily = policy.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        BudgetUsageSnapshot(
            spent_today_usd=Decimal("0.09"),
            spent_week_usd=Decimal("0.09"),
            estimated_cost_usd=Decimal("0.02"),
        ),
    )
    weekly = policy.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        BudgetUsageSnapshot(
            spent_today_usd=Decimal("0.01"),
            spent_week_usd=Decimal("0.19"),
            estimated_cost_usd=Decimal("0.02"),
        ),
    )

    assert daily.decision is BudgetDecisionType.BLOCK
    assert daily.reason == "daily_budget_exceeded"
    assert any("дневной лимит" in line for line in daily.status_lines)
    assert weekly.decision is BudgetDecisionType.BLOCK
    assert weekly.reason == "weekly_budget_exceeded"
    assert any("недельный лимит" in line for line in weekly.status_lines)


def test_coding_route_uses_code_agent_settings_and_self_coding_budget() -> None:
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetUsageSnapshot,
    )

    policy = AgentBudgetPolicy.from_settings(
        _settings(
            AUTONOMOUS_SELF_CODING_ENABLED="true",
            CODE_AGENT_MODEL="gpt-5.5",
            CODE_AGENT_REASONING_EFFORT="xhigh",
            AUTONOMOUS_SELF_CODING_BUDGET_DAILY_USD="2.50",
            AUTONOMOUS_SELF_CODING_BUDGET_WEEKLY_USD="7.50",
        )
    )

    decision = policy.evaluate(
        BudgetJobKind.CODING,
        BudgetUsageSnapshot(estimated_cost_usd=Decimal("1.25")),
    )

    assert decision.decision is BudgetDecisionType.ALLOW
    assert decision.route.provider == "codex_cli"
    assert decision.route.model == "gpt-5.5"
    assert decision.route.reasoning_effort == "xhigh"
    assert decision.route.daily_budget_usd == Decimal("2.5")
    assert decision.route.weekly_budget_usd == Decimal("7.5")
    assert decision.route.auto_run_allowed is True


def test_tier3_review_requires_nikita_even_under_budget() -> None:
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetUsageSnapshot,
    )

    policy = AgentBudgetPolicy.from_settings(_settings())

    decision = policy.evaluate(
        BudgetJobKind.TIER3_REVIEW,
        BudgetUsageSnapshot(estimated_cost_usd=Decimal("0.01")),
    )

    assert decision.decision is BudgetDecisionType.ASK_NIKITA
    assert decision.route.auto_run_allowed is False
    assert decision.route.requires_nikita is True
    assert any("Tier 3" in line for line in decision.status_lines)


def test_budget_usage_ledger_builds_daily_weekly_snapshot(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime, timedelta

    from src.agent_runtime.budget_policy import (
        BudgetJobKind,
        BudgetUsageRecord,
        FileBudgetUsageLedger,
    )

    now = datetime(2026, 5, 14, 12, tzinfo=UTC)
    ledger = FileBudgetUsageLedger(tmp_path / "budget-usage.jsonl")
    ledger.record(
        BudgetUsageRecord(
            job_kind=BudgetJobKind.READONLY_RESEARCH,
            cost_usd=Decimal("0.04"),
            occurred_at=now - timedelta(hours=1),
            job_id="job-today",
        )
    )
    ledger.record(
        BudgetUsageRecord(
            job_kind=BudgetJobKind.SUMMARIZATION,
            cost_usd=Decimal("0.06"),
            occurred_at=now - timedelta(days=2),
            job_id="job-week",
        )
    )
    ledger.record(
        BudgetUsageRecord(
            job_kind=BudgetJobKind.CODING,
            cost_usd=Decimal("1.00"),
            occurred_at=now - timedelta(hours=1),
            job_id="job-coding",
        )
    )

    snapshot = ledger.snapshot_for(
        BudgetJobKind.READONLY_RESEARCH,
        estimated_cost_usd=Decimal("0.02"),
        now=now,
    )
    coding_snapshot = ledger.snapshot_for(
        BudgetJobKind.CODING,
        estimated_cost_usd=Decimal("0.50"),
        now=now,
    )

    assert snapshot.spent_today_usd == Decimal("0.04")
    assert snapshot.spent_week_usd == Decimal("0.10")
    assert snapshot.estimated_cost_usd == Decimal("0.02")
    assert coding_snapshot.spent_today_usd == Decimal("1.00")
    assert coding_snapshot.spent_week_usd == Decimal("1.00")


def test_budget_preflight_gate_uses_recorded_usage_before_starting_job(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetPreflightGate,
        BudgetUsageRecord,
        FileBudgetUsageLedger,
    )

    now = datetime(2026, 5, 14, 12, tzinfo=UTC)
    policy = AgentBudgetPolicy.from_settings(
        _settings(
            AUTONOMOUS_LOOP_BUDGET_DAILY_USD="0.10",
            AUTONOMOUS_LOOP_BUDGET_WEEKLY_USD="1.00",
        )
    )
    ledger = FileBudgetUsageLedger(tmp_path / "budget-usage.jsonl")
    ledger.record(
        BudgetUsageRecord(
            job_kind=BudgetJobKind.MEMORY_STAGING,
            cost_usd=Decimal("0.09"),
            occurred_at=now,
            job_id="job-existing",
        )
    )
    gate = BudgetPreflightGate(policy=policy, ledger=ledger)

    decision = gate.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        estimated_cost_usd=Decimal("0.02"),
        now=now,
    )

    assert decision.decision is BudgetDecisionType.BLOCK
    assert decision.reason == "daily_budget_exceeded"


def test_budget_policy_decision_status_report_explains_block_without_execution() -> (
    None
):
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetDecisionType,
        BudgetJobKind,
        BudgetUsageSnapshot,
        render_budget_policy_decision_status,
    )

    policy = AgentBudgetPolicy.from_settings(
        _settings(
            AUTONOMOUS_LOOP_BUDGET_DAILY_USD="0.10",
            AUTONOMOUS_LOOP_BUDGET_WEEKLY_USD="1.00",
        )
    )
    decision = policy.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        BudgetUsageSnapshot(
            spent_today_usd=Decimal("0.09"),
            spent_week_usd=Decimal("0.09"),
            estimated_cost_usd=Decimal("0.02"),
        ),
    )

    status = render_budget_policy_decision_status(decision)

    assert decision.decision is BudgetDecisionType.BLOCK
    assert "Budget policy: block" in status
    assert "job_kind: readonly_research" in status
    assert "reason: daily_budget_exceeded" in status
    assert "route: codex_cli/default" in status
    assert "effort: medium" in status
    assert "auto_run_allowed: yes" in status
    assert "execution: not_started" in status
    assert "Остановлено: дневной лимит" in status


def test_budget_policy_decision_status_report_marks_nikita_required() -> None:
    from src.agent_runtime.budget_policy import (
        AgentBudgetPolicy,
        BudgetJobKind,
        BudgetUsageSnapshot,
        render_budget_policy_decision_status,
    )

    policy = AgentBudgetPolicy.from_settings(_settings())
    decision = policy.evaluate(
        BudgetJobKind.TIER3_REVIEW,
        BudgetUsageSnapshot(estimated_cost_usd=Decimal("0.01")),
    )

    status = render_budget_policy_decision_status(decision)

    assert "Budget policy: ask_nikita" in status
    assert "job_kind: tier3_review" in status
    assert "requires_nikita: yes" in status
    assert "execution: not_started" in status


def test_resource_autonomy_policy_allows_only_day_window() -> None:
    from src.agent_runtime.budget_policy import (
        AutonomyResourceLimitProfile,
        AutonomyResourcePolicy,
        AutonomyResourceUsageSnapshot,
        BudgetDecisionType,
        BudgetJobKind,
    )

    moscow = timezone(timedelta(hours=3))
    policy = AutonomyResourcePolicy(
        profiles={
            BudgetJobKind.READONLY_RESEARCH: AutonomyResourceLimitProfile(
                job_kind=BudgetJobKind.READONLY_RESEARCH,
                window_start_hour=10,
                window_duration_hours=12,
                max_jobs_per_window=12,
                max_runtime_seconds_per_window=12 * 60 * 60,
                max_retries_per_fingerprint=2,
                max_concurrent_jobs=1,
                auto_run_allowed=True,
            )
        }
    )

    day = policy.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        AutonomyResourceUsageSnapshot(now=datetime(2026, 5, 14, 12, tzinfo=moscow)),
    )
    night = policy.evaluate(
        BudgetJobKind.READONLY_RESEARCH,
        AutonomyResourceUsageSnapshot(now=datetime(2026, 5, 14, 23, tzinfo=moscow)),
    )

    assert day.decision is BudgetDecisionType.ALLOW
    assert day.reason == "within_resource_limits"
    assert night.decision is BudgetDecisionType.BLOCK
    assert night.reason == "outside_day_window"
    assert any("10:00-22:00" in line for line in night.status_lines)


def test_resource_autonomy_policy_blocks_jobs_runtime_retries_and_concurrency() -> None:
    from src.agent_runtime.budget_policy import (
        AutonomyResourceLimitProfile,
        AutonomyResourcePolicy,
        AutonomyResourceUsageSnapshot,
        BudgetDecisionType,
        BudgetJobKind,
    )

    profile = AutonomyResourceLimitProfile(
        job_kind=BudgetJobKind.SPEC_WRITING,
        window_start_hour=10,
        window_duration_hours=12,
        max_jobs_per_window=2,
        max_runtime_seconds_per_window=120,
        max_retries_per_fingerprint=1,
        max_concurrent_jobs=1,
        auto_run_allowed=True,
    )
    policy = AutonomyResourcePolicy(profiles={BudgetJobKind.SPEC_WRITING: profile})
    now = datetime(2026, 5, 14, 12, tzinfo=UTC)

    too_many_jobs = policy.evaluate(
        BudgetJobKind.SPEC_WRITING,
        AutonomyResourceUsageSnapshot(jobs_started_in_window=2, now=now),
    )
    too_much_runtime = policy.evaluate(
        BudgetJobKind.SPEC_WRITING,
        AutonomyResourceUsageSnapshot(runtime_seconds_in_window=121, now=now),
    )
    too_many_retries = policy.evaluate(
        BudgetJobKind.SPEC_WRITING,
        AutonomyResourceUsageSnapshot(retries_for_fingerprint=1, now=now),
    )
    too_many_active = policy.evaluate(
        BudgetJobKind.SPEC_WRITING,
        AutonomyResourceUsageSnapshot(active_jobs=1, now=now),
    )

    assert too_many_jobs.decision is BudgetDecisionType.BLOCK
    assert too_many_jobs.reason == "job_count_limit_exceeded"
    assert too_much_runtime.decision is BudgetDecisionType.BLOCK
    assert too_much_runtime.reason == "runtime_limit_exceeded"
    assert too_many_retries.decision is BudgetDecisionType.BLOCK
    assert too_many_retries.reason == "retry_limit_exceeded"
    assert too_many_active.decision is BudgetDecisionType.BLOCK
    assert too_many_active.reason == "concurrency_limit_exceeded"
