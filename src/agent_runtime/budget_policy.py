"""Dry-run model, effort and budget policy for autonomous Agent Runtime work."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.agent_runtime.models import AgentJob, ContextCapsule
    from src.core.config import Settings

TierValue = Literal["worker", "analyst", "strategist"]
ReasoningEffortValue = Literal["low", "medium", "high", "xhigh"]


class BudgetJobKind(StrEnum):
    """Stable job kinds covered by the autonomous budget policy."""

    CHAT = "chat"
    READONLY_RESEARCH = "readonly_research"
    SUMMARIZATION = "summarization"
    MEMORY_STAGING = "memory_staging"
    SOCIAL_JUDGEMENT = "social_judgement"
    SPEC_WRITING = "spec_writing"
    CODING = "coding"
    TIER3_REVIEW = "tier3_review"


class BudgetDecisionType(StrEnum):
    """Budget policy outcome."""

    ALLOW = "allow"
    BLOCK = "block"
    ASK_NIKITA = "ask_nikita"


class BudgetUsageSnapshot(BaseModel):
    """Already spent budget plus the estimate for one planned job."""

    spent_today_usd: Decimal = Field(default=Decimal("0"), ge=0)
    spent_week_usd: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)


class BudgetUsageRecord(BaseModel):
    """One completed cost observation from an autonomous/runtime job."""

    model_config = ConfigDict(extra="ignore")

    job_kind: BudgetJobKind
    cost_usd: Decimal = Field(ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    job_id: str = ""
    source: str = ""

    @field_validator("occurred_at")
    @classmethod
    def _normalize_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class BudgetRoute(BaseModel):
    """Provider/model/effort and caps selected for a job kind."""

    job_kind: BudgetJobKind
    tier: TierValue
    provider: str
    model: str
    reasoning_effort: ReasoningEffortValue
    daily_budget_usd: Decimal
    weekly_budget_usd: Decimal
    auto_run_allowed: bool
    requires_nikita: bool = False
    notes: tuple[str, ...] = ()


class BudgetPolicyDecision(BaseModel):
    """Dry-run budget decision returned to Жвуша for status/explanation."""

    decision: BudgetDecisionType
    route: BudgetRoute
    reason: str
    status_lines: tuple[str, ...]


class AutonomyResourceUsageSnapshot(BaseModel):
    """Observed resource usage for one daylight autonomy window."""

    jobs_started_in_window: int = Field(default=0, ge=0)
    runtime_seconds_in_window: int = Field(default=0, ge=0)
    retries_for_fingerprint: int = Field(default=0, ge=0)
    active_jobs: int = Field(default=0, ge=0)
    now: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("now")
    @classmethod
    def _normalize_now(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class AutonomyResourceLimitProfile(BaseModel):
    """Hard resource/autonomy caps for one class of autonomous work."""

    job_kind: BudgetJobKind
    window_start_hour: int = Field(ge=0, le=23)
    window_duration_hours: int = Field(ge=1, le=24)
    max_jobs_per_window: int = Field(ge=0)
    max_runtime_seconds_per_window: int = Field(ge=0)
    max_retries_per_fingerprint: int = Field(ge=0)
    max_concurrent_jobs: int = Field(ge=0)
    auto_run_allowed: bool
    requires_nikita: bool = False
    notes: tuple[str, ...] = ()


class AutonomyResourcePolicyDecision(BaseModel):
    """Resource/autonomy policy outcome without cost semantics."""

    decision: BudgetDecisionType
    profile: AutonomyResourceLimitProfile
    reason: str
    status_lines: tuple[str, ...]


class AutonomyResourcePolicy:
    """Hard caps for autonomous loops, paced across a daylight window."""

    def __init__(
        self,
        profiles: Mapping[BudgetJobKind, AutonomyResourceLimitProfile],
    ) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def from_settings(cls, settings: Settings) -> AutonomyResourcePolicy:
        window_start = settings.autonomous_day_window_start_hour
        window_duration = settings.autonomous_day_window_duration_hours
        loop_max_jobs = settings.autonomous_loop_max_jobs_per_day
        loop_max_runtime = settings.autonomous_loop_max_runtime_seconds_per_day
        loop_max_retries = settings.autonomous_loop_max_retries_per_job
        loop_max_concurrent = settings.autonomous_loop_max_concurrent_jobs
        profiles = {
            BudgetJobKind.CHAT: _resource_profile(
                job_kind=BudgetJobKind.CHAT,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=False,
                notes=("Обычный чат реактивный и не является autonomous loop.",),
            ),
            BudgetJobKind.READONLY_RESEARCH: _resource_profile(
                job_kind=BudgetJobKind.READONLY_RESEARCH,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=True,
            ),
            BudgetJobKind.SUMMARIZATION: _resource_profile(
                job_kind=BudgetJobKind.SUMMARIZATION,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=True,
            ),
            BudgetJobKind.MEMORY_STAGING: _resource_profile(
                job_kind=BudgetJobKind.MEMORY_STAGING,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=True,
            ),
            BudgetJobKind.SOCIAL_JUDGEMENT: _resource_profile(
                job_kind=BudgetJobKind.SOCIAL_JUDGEMENT,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=True,
                notes=("Social judgement stays grant/approval-gated.",),
            ),
            BudgetJobKind.SPEC_WRITING: _resource_profile(
                job_kind=BudgetJobKind.SPEC_WRITING,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=True,
                notes=("Spec writing drafts safety contracts; gates remain separate.",),
            ),
            BudgetJobKind.CODING: _resource_profile(
                job_kind=BudgetJobKind.CODING,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=settings.autonomous_self_coding_max_jobs_per_day,
                max_runtime_seconds_per_window=(
                    settings.autonomous_self_coding_max_runtime_seconds_per_day
                ),
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=settings.code_agent_max_concurrent,
                auto_run_allowed=settings.autonomous_self_coding_enabled,
                notes=("Coding still requires approved spec and self-coding gates.",),
            ),
            BudgetJobKind.TIER3_REVIEW: _resource_profile(
                job_kind=BudgetJobKind.TIER3_REVIEW,
                window_start_hour=window_start,
                window_duration_hours=window_duration,
                max_jobs_per_window=loop_max_jobs,
                max_runtime_seconds_per_window=loop_max_runtime,
                max_retries_per_fingerprint=loop_max_retries,
                max_concurrent_jobs=loop_max_concurrent,
                auto_run_allowed=False,
                requires_nikita=True,
                notes=("Tier 3 review requires Никита, never autonomous approval.",),
            ),
        }
        return cls(profiles)

    def profile_for(self, job_kind: BudgetJobKind) -> AutonomyResourceLimitProfile:
        """Return configured resource caps for a job kind."""
        return self._profiles[job_kind]

    def evaluate(
        self,
        job_kind: BudgetJobKind,
        usage: AutonomyResourceUsageSnapshot | None = None,
    ) -> AutonomyResourcePolicyDecision:
        """Check daylight window, jobs, runtime, retries and concurrency caps."""
        profile = self.profile_for(job_kind)
        usage = usage or AutonomyResourceUsageSnapshot()
        status_lines = _resource_status_lines(profile, usage)
        if profile.requires_nikita:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.ASK_NIKITA,
                profile=profile,
                reason="requires_nikita",
                status_lines=(
                    *status_lines,
                    "Tier 3 не запускается автономно: нужен Никита.",
                ),
            )
        if not _inside_day_window(
            usage.now,
            start_hour=profile.window_start_hour,
            duration_hours=profile.window_duration_hours,
        ):
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="outside_day_window",
                status_lines=(
                    *status_lines,
                    "Остановлено: autonomous loops разрешены только в дневном окне.",
                ),
            )
        if not profile.auto_run_allowed:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="auto_run_disabled",
                status_lines=(*status_lines, "Остановлено: auto-run выключен."),
            )
        if usage.active_jobs >= profile.max_concurrent_jobs:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="concurrency_limit_exceeded",
                status_lines=(
                    *status_lines,
                    "Остановлено: лимит параллельных autonomous jobs исчерпан.",
                ),
            )
        if usage.jobs_started_in_window >= profile.max_jobs_per_window:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="job_count_limit_exceeded",
                status_lines=(
                    *status_lines,
                    "Остановлено: лимит jobs за окно исчерпан.",
                ),
            )
        if usage.runtime_seconds_in_window >= profile.max_runtime_seconds_per_window:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="runtime_limit_exceeded",
                status_lines=(*status_lines, "Остановлено: runtime за окно исчерпан."),
            )
        if usage.retries_for_fingerprint >= profile.max_retries_per_fingerprint:
            return AutonomyResourcePolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                profile=profile,
                reason="retry_limit_exceeded",
                status_lines=(*status_lines, "Остановлено: retry limit исчерпан."),
            )
        return AutonomyResourcePolicyDecision(
            decision=BudgetDecisionType.ALLOW,
            profile=profile,
            reason="within_resource_limits",
            status_lines=(*status_lines, "Разрешено resource/autonomy policy."),
        )


class AgentBudgetPolicy:
    """Deterministic policy matrix for autonomous model/cost routing."""

    def __init__(self, routes: Mapping[BudgetJobKind, BudgetRoute]) -> None:
        self._routes = dict(routes)

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentBudgetPolicy:
        loop_daily = _money(settings.autonomous_loop_budget_daily_usd)
        loop_weekly = _money(settings.autonomous_loop_budget_weekly_usd)
        coding_daily = _money(settings.autonomous_self_coding_budget_daily_usd)
        coding_weekly = _money(settings.autonomous_self_coding_budget_weekly_usd)

        routes = {
            BudgetJobKind.CHAT: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.CHAT,
                tier=settings.chat_assistant_tier,
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=False,
                notes=("Обычный чат реактивный и не является автономным loop.",),
            ),
            BudgetJobKind.READONLY_RESEARCH: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.READONLY_RESEARCH,
                tier="worker",
                effort="medium",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=True,
            ),
            BudgetJobKind.SUMMARIZATION: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.SUMMARIZATION,
                tier="worker",
                effort="low",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=True,
            ),
            BudgetJobKind.MEMORY_STAGING: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.MEMORY_STAGING,
                tier="worker",
                effort="low",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=True,
            ),
            BudgetJobKind.SOCIAL_JUDGEMENT: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.SOCIAL_JUDGEMENT,
                tier="analyst",
                effort="high",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=True,
                notes=("Социальное решение дороже summary, потому что есть риск.",),
            ),
            BudgetJobKind.SPEC_WRITING: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.SPEC_WRITING,
                tier="strategist",
                effort="high",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=True,
                notes=(
                    "Spec writing drafts safety contract; approval gates stay separate.",
                ),
            ),
            BudgetJobKind.CODING: BudgetRoute(
                job_kind=BudgetJobKind.CODING,
                tier="strategist",
                provider=settings.code_agent_backend,
                model=settings.code_agent_model or settings.strategist_model,
                reasoning_effort=settings.code_agent_reasoning_effort,
                daily_budget_usd=coding_daily,
                weekly_budget_usd=coding_weekly,
                auto_run_allowed=settings.autonomous_self_coding_enabled,
                notes=(
                    "Coding still requires approved spec and existing self-coding gates.",
                ),
            ),
            BudgetJobKind.TIER3_REVIEW: _tier_route(
                settings=settings,
                job_kind=BudgetJobKind.TIER3_REVIEW,
                tier="strategist",
                effort="xhigh",
                daily_budget_usd=loop_daily,
                weekly_budget_usd=loop_weekly,
                auto_run_allowed=False,
                requires_nikita=True,
                notes=(
                    "Tier 3 review is a Никита decision, not an auto budget decision.",
                ),
            ),
        }
        return cls(routes)

    def route_for(self, job_kind: BudgetJobKind) -> BudgetRoute:
        """Return the configured route for a job kind."""
        return self._routes[job_kind]

    def evaluate(
        self,
        job_kind: BudgetJobKind,
        usage: BudgetUsageSnapshot | None = None,
    ) -> BudgetPolicyDecision:
        """Dry-run whether the estimated job fits policy and budget caps."""
        route = self.route_for(job_kind)
        usage = usage or BudgetUsageSnapshot()
        status_lines = _status_lines(route, usage)
        if route.requires_nikita:
            return BudgetPolicyDecision(
                decision=BudgetDecisionType.ASK_NIKITA,
                route=route,
                reason="requires_nikita",
                status_lines=(
                    *status_lines,
                    "Tier 3 review не запускается автономно: нужен Никита.",
                ),
            )
        if usage.spent_today_usd + usage.estimated_cost_usd > route.daily_budget_usd:
            return BudgetPolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                route=route,
                reason="daily_budget_exceeded",
                status_lines=(
                    *status_lines,
                    "Остановлено: дневной лимит автономного бюджета будет превышен.",
                ),
            )
        if usage.spent_week_usd + usage.estimated_cost_usd > route.weekly_budget_usd:
            return BudgetPolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                route=route,
                reason="weekly_budget_exceeded",
                status_lines=(
                    *status_lines,
                    "Остановлено: недельный лимит автономного бюджета будет превышен.",
                ),
            )
        return BudgetPolicyDecision(
            decision=BudgetDecisionType.ALLOW,
            route=route,
            reason="within_budget",
            status_lines=(*status_lines, "Разрешено budget policy dry-run."),
        )

    def dry_run_report(
        self,
        usage_by_kind: Mapping[BudgetJobKind, BudgetUsageSnapshot] | None = None,
    ) -> str:
        """Render a compact operator-readable report without executing jobs."""
        usage_by_kind = usage_by_kind or {}
        lines = ["## Budget policy dry-run"]
        for kind in BudgetJobKind:
            decision = self.evaluate(kind, usage_by_kind.get(kind))
            route = decision.route
            lines.append(
                "- "
                f"{kind.value}: {decision.decision.value}, "
                f"{route.provider}/{route.model}, effort={route.reasoning_effort}, "
                f"reason={decision.reason}"
            )
        return "\n".join(lines)


class FileBudgetUsageLedger:
    """Append-only JSONL usage sink for autonomous budget preflight."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, record: BudgetUsageRecord) -> None:
        """Append one observed usage event."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")

    def list_records(self) -> tuple[BudgetUsageRecord, ...]:
        """Load usage records, ignoring corrupt lines instead of blocking preflight."""
        if not self._path.is_file():
            return ()
        records: list[BudgetUsageRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(BudgetUsageRecord.model_validate_json(line))
            except ValueError:
                continue
        return tuple(records)

    def snapshot_for(
        self,
        job_kind: BudgetJobKind,
        *,
        estimated_cost_usd: Decimal,
        now: datetime | None = None,
    ) -> BudgetUsageSnapshot:
        """Return current spent_today/spent_week for this job's budget bucket."""
        current = _normalize_dt(now or datetime.now(UTC))
        bucket = _budget_bucket(job_kind)
        spent_today = Decimal("0")
        spent_week = Decimal("0")
        for record in self.list_records():
            if _budget_bucket(record.job_kind) != bucket:
                continue
            occurred = _normalize_dt(record.occurred_at)
            if occurred.date() == current.date():
                spent_today += record.cost_usd
            if occurred.isocalendar()[:2] == current.isocalendar()[:2]:
                spent_week += record.cost_usd
        return BudgetUsageSnapshot(
            spent_today_usd=spent_today,
            spent_week_usd=spent_week,
            estimated_cost_usd=estimated_cost_usd,
        )


class BudgetPreflightGate:
    """Evaluate recorded usage before starting an autonomous job."""

    def __init__(
        self,
        *,
        policy: AgentBudgetPolicy,
        ledger: FileBudgetUsageLedger,
    ) -> None:
        self._policy = policy
        self._ledger = ledger

    def evaluate(
        self,
        job_kind: BudgetJobKind,
        *,
        estimated_cost_usd: Decimal,
        now: datetime | None = None,
    ) -> BudgetPolicyDecision:
        """Return the policy decision using recorded usage for this budget bucket."""
        usage = self._ledger.snapshot_for(
            job_kind,
            estimated_cost_usd=estimated_cost_usd,
            now=now,
        )
        return self._policy.evaluate(job_kind, usage)


class AgentRuntimeBudgetUsageRecorder:
    """Record completed Agent Runtime jobs with budget metadata into a ledger."""

    def __init__(
        self,
        *,
        ledger: FileBudgetUsageLedger,
        source: str = "agent_runtime",
    ) -> None:
        self._ledger = ledger
        self._source = source

    async def record_completed_job(
        self,
        *,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> int:
        """Record budget usage after a job completed successfully."""
        del capsule
        metadata = job.context_pack.metadata
        job_kind = _metadata_job_kind(metadata)
        if job_kind is None:
            return 0
        cost = _metadata_cost(metadata)
        if cost is None:
            return 0
        self._ledger.record(
            BudgetUsageRecord(
                job_kind=job_kind,
                cost_usd=cost,
                job_id=job.id,
                source=metadata.get("budget_source", self._source),
            )
        )
        return 1


def render_budget_policy_decision_status(decision: BudgetPolicyDecision) -> str:
    """Render an operator-facing budget preflight status without running jobs."""
    route = decision.route
    lines = [
        f"Budget policy: {decision.decision.value}",
        f"job_kind: {route.job_kind.value}",
        f"reason: {decision.reason}",
        f"route: {route.provider}/{route.model}",
        f"tier: {route.tier}",
        f"effort: {route.reasoning_effort}",
        f"daily_budget_usd: {route.daily_budget_usd:.2f}",
        f"weekly_budget_usd: {route.weekly_budget_usd:.2f}",
        f"auto_run_allowed: {_yes_no(route.auto_run_allowed)}",
        f"requires_nikita: {_yes_no(route.requires_nikita)}",
    ]
    if route.notes:
        lines.append(f"notes: {'; '.join(route.notes)}")
    if decision.status_lines:
        lines.append("status:")
        lines.extend(f"- {line}" for line in decision.status_lines)
    lines.append("execution: not_started")
    return "\n".join(lines)


def _tier_route(
    *,
    settings: Settings,
    job_kind: BudgetJobKind,
    tier: TierValue,
    daily_budget_usd: Decimal,
    weekly_budget_usd: Decimal,
    auto_run_allowed: bool,
    effort: ReasoningEffortValue | None = None,
    requires_nikita: bool = False,
    notes: tuple[str, ...] = (),
) -> BudgetRoute:
    return BudgetRoute(
        job_kind=job_kind,
        tier=tier,
        provider=_provider_for_tier(settings, tier),
        model=_model_for_tier(settings, tier),
        reasoning_effort=effort or _effort_for_tier(settings, tier),
        daily_budget_usd=daily_budget_usd,
        weekly_budget_usd=weekly_budget_usd,
        auto_run_allowed=auto_run_allowed,
        requires_nikita=requires_nikita,
        notes=notes,
    )


def _resource_profile(
    *,
    job_kind: BudgetJobKind,
    window_start_hour: int,
    window_duration_hours: int,
    max_jobs_per_window: int,
    max_runtime_seconds_per_window: int,
    max_retries_per_fingerprint: int,
    max_concurrent_jobs: int,
    auto_run_allowed: bool,
    requires_nikita: bool = False,
    notes: tuple[str, ...] = (),
) -> AutonomyResourceLimitProfile:
    return AutonomyResourceLimitProfile(
        job_kind=job_kind,
        window_start_hour=window_start_hour,
        window_duration_hours=window_duration_hours,
        max_jobs_per_window=max_jobs_per_window,
        max_runtime_seconds_per_window=max_runtime_seconds_per_window,
        max_retries_per_fingerprint=max_retries_per_fingerprint,
        max_concurrent_jobs=max_concurrent_jobs,
        auto_run_allowed=auto_run_allowed,
        requires_nikita=requires_nikita,
        notes=notes,
    )


def _provider_for_tier(settings: Settings, tier: TierValue) -> str:
    return {
        "worker": settings.worker_provider,
        "analyst": settings.analyst_provider,
        "strategist": settings.strategist_provider,
    }[tier]


def _model_for_tier(settings: Settings, tier: TierValue) -> str:
    return {
        "worker": settings.worker_model,
        "analyst": settings.analyst_model,
        "strategist": settings.strategist_model,
    }[tier]


def _effort_for_tier(settings: Settings, tier: TierValue) -> ReasoningEffortValue:
    return {
        "worker": settings.worker_reasoning_effort,
        "analyst": settings.analyst_reasoning_effort,
        "strategist": settings.strategist_reasoning_effort,
    }[tier]


def _money(value: float) -> Decimal:
    return Decimal(str(value))


def _budget_bucket(job_kind: BudgetJobKind) -> str:
    if job_kind is BudgetJobKind.CODING:
        return "self_coding"
    return "autonomous_loop"


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _metadata_job_kind(metadata: Mapping[str, str]) -> BudgetJobKind | None:
    raw = metadata.get("budget_job_kind", "").strip()
    if not raw:
        return None
    try:
        return BudgetJobKind(raw)
    except ValueError:
        return None


def _metadata_cost(metadata: Mapping[str, str]) -> Decimal | None:
    raw = (
        metadata.get("budget_observed_cost_usd", "").strip()
        or metadata.get("budget_estimated_cost_usd", "").strip()
    )
    if not raw:
        return None
    try:
        cost = Decimal(raw)
    except InvalidOperation:
        return None
    if cost < 0:
        return None
    return cost


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _status_lines(route: BudgetRoute, usage: BudgetUsageSnapshot) -> tuple[str, ...]:
    projected_today = usage.spent_today_usd + usage.estimated_cost_usd
    projected_week = usage.spent_week_usd + usage.estimated_cost_usd
    return (
        (
            f"route: {route.job_kind.value} -> {route.provider}/{route.model}, "
            f"effort={route.reasoning_effort}"
        ),
        (
            f"дневной лимит: ${projected_today:.2f}/${route.daily_budget_usd:.2f}; "
            f"недельный лимит: ${projected_week:.2f}/${route.weekly_budget_usd:.2f}"
        ),
    )


def _inside_day_window(
    value: datetime,
    *,
    start_hour: int,
    duration_hours: int,
) -> bool:
    hour = value.hour + (value.minute / 60) + (value.second / 3600)
    end_hour = (start_hour + duration_hours) % 24
    if duration_hours >= 24:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _day_window_label(profile: AutonomyResourceLimitProfile) -> str:
    end_hour = (profile.window_start_hour + profile.window_duration_hours) % 24
    return f"{profile.window_start_hour:02d}:00-{end_hour:02d}:00"


def _resource_status_lines(
    profile: AutonomyResourceLimitProfile,
    usage: AutonomyResourceUsageSnapshot,
) -> tuple[str, ...]:
    return (
        (
            f"day_window: {_day_window_label(profile)} "
            f"({profile.window_duration_hours}h)"
        ),
        (
            "jobs: "
            f"{usage.jobs_started_in_window}/{profile.max_jobs_per_window}; "
            "runtime_seconds: "
            f"{usage.runtime_seconds_in_window}/"
            f"{profile.max_runtime_seconds_per_window}; "
            f"active_jobs: {usage.active_jobs}/{profile.max_concurrent_jobs}; "
            "retries: "
            f"{usage.retries_for_fingerprint}/"
            f"{profile.max_retries_per_fingerprint}"
        ),
    )
