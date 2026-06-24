"""Durable value models for Жвуша's LifeRuntime."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LifeEventKind(StrEnum):
    """Input event kinds observed by LifeRuntime."""

    USER_MESSAGE = "user_message"
    SILENCE_TICK = "silence_tick"
    SCHEDULE_TICK = "schedule_tick"
    DESIRE_STALE = "desire_stale"
    AFFECT_SHIFT = "affect_shift"
    HOMEOSTASIS_DRIFT = "homeostasis_drift"
    AGENT_JOB_RESULT = "agent_job_result"
    MEMORY_CANDIDATE_READY = "memory_candidate_ready"
    WORKSPACE_SIGNAL = "workspace_signal"
    FAILED_JOB = "failed_job"
    BUDGET_GUARD = "budget_guard"


class LifeEventSource(StrEnum):
    """Subsystem that emitted a LifeEvent."""

    DAEMON = "daemon"
    BOT = "bot"
    AGENT_RUNTIME = "agent_runtime"
    MEMORY = "memory"
    PERSONALITY = "personality"
    WORKSPACE = "workspace"
    SYSTEM = "system"


class LifePriority(StrEnum):
    """Priority for attention scoring."""

    CRITICAL = "critical"
    NORMAL = "normal"
    BACKGROUND = "background"


class SelfStateMode(StrEnum):
    """Durable LifeRuntime mode."""

    IDLE = "idle"
    ATTENDING = "attending"
    REFLECTING = "reflecting"
    WAITING_USER = "waiting_user"
    WAITING_APPROVAL = "waiting_approval"
    COOLDOWN = "cooldown"
    RECOVERING = "recovering"


class AttentionStatus(StrEnum):
    """Attention item lifecycle status."""

    NEW = "new"
    SELECTED = "selected"
    DEFERRED = "deferred"
    RESOLVED = "resolved"
    BLOCKED = "blocked"


class InnerDecisionType(StrEnum):
    """Allowed bounded decisions emitted by one life tick."""

    THINK = "think"
    REFLECT = "reflect"
    STAGE_MEMORY = "stage_memory"
    PROPOSE_AGENT_JOB = "propose_agent_job"
    ASK_NIKITA = "ask_nikita"
    DEFER = "defer"
    IGNORE = "ignore"
    SLEEP = "sleep"
    WAIT_FOR_APPROVAL = "wait_for_approval"


class LifeActionRequestKind(StrEnum):
    """Safe handoff kinds from LifeRuntime to existing gates."""

    AGENT_RUNTIME_JOB = "agent_runtime_job"
    MEMORY_STAGING = "memory_staging"
    APPROVAL_REQUEST = "approval_request"


class LifeEvent(BaseModel):
    """Append-only input event observed by LifeRuntime."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    kind: LifeEventKind
    source: LifeEventSource
    priority: LifePriority = LifePriority.NORMAL
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str = ""
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("id", "dedupe_key", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()


class SelfState(BaseModel):
    """Durable snapshot of Жвуша's internal continuity."""

    schema_version: int = 1
    state_id: str = Field(default_factory=lambda: f"self-state-{uuid4().hex}")
    mode: SelfStateMode = SelfStateMode.IDLE
    current_focus: str = ""
    open_loops: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    active_desires: tuple[str, ...] = ()
    affect_summary: str = ""
    homeostasis_summary: str = ""
    attention_summary: str = ""
    recent_decision_ids: tuple[str, ...] = ()
    budget_state: str = "normal"
    last_tick_id: str = ""
    last_tick_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AttentionItem(BaseModel):
    """Candidate target selected for a bounded life tick."""

    id: str
    event_id: str
    summary: str
    status: AttentionStatus = AttentionStatus.NEW
    salience: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    relation_to_nikita: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0, le=1.0)
    decay_after: datetime
    evidence: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DriveVector(BaseModel):
    """Personality-derived deterministic policy vector for MVP scoring."""

    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    care_for_nikita: float = Field(default=0.7, ge=0.0, le=1.0)
    honesty_pressure: float = Field(default=0.7, ge=0.0, le=1.0)
    caution: float = Field(default=0.7, ge=0.0, le=1.0)
    complexity_growth: float = Field(default=0.5, ge=0.0, le=1.0)
    relational_continuity: float = Field(default=0.7, ge=0.0, le=1.0)
    energy_budget: float = Field(default=0.7, ge=0.0, le=1.0)
    learning_pressure: float = Field(default=0.5, ge=0.0, le=1.0)
    action_pressure: float = Field(default=0.2, ge=0.0, le=1.0)
    silence_pressure: float = Field(default=0.4, ge=0.0, le=1.0)


class LifeActionRequest(BaseModel):
    """Safe handoff object to Agent Runtime, staging or approval stores."""

    id: str = Field(default_factory=lambda: f"life-action-{uuid4().hex}")
    requested_by_tick_id: str = ""
    kind: LifeActionRequestKind
    profile_id: str = ""
    capabilities_requested: tuple[str, ...] = ()
    denied_capabilities: tuple[str, ...] = ()
    reason: str = ""
    requires_approval: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InnerDecision(BaseModel):
    """One bounded decision emitted by a LifeRuntime tick."""

    id: str = Field(default_factory=lambda: f"life-decision-{uuid4().hex}")
    decision_type: InnerDecisionType
    reason: str
    action_request: LifeActionRequest | None = None
    requires_approval: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LifeRuntimeSafetyVerdict(BaseModel):
    """Safety result attached to each LifeTick."""

    allowed: bool
    reason: str
    denied_capabilities: tuple[str, ...] = ()


class LifeTick(BaseModel):
    """Auditable unit of bounded LifeRuntime execution."""

    id: str
    trigger_event_id: str
    started_at: datetime
    finished_at: datetime
    loaded_state_hash: str
    selected_attention_id: str
    drive_vector: DriveVector
    decision: InnerDecision
    safety_verdict: LifeRuntimeSafetyVerdict
    agent_job_id: str = ""
    reflection_capsule_id: str = ""
    state_delta: dict[str, str] = Field(default_factory=dict)
    result_summary: str = ""
    error: str = ""


class ReflectionCapsule(BaseModel):
    """Internal reflection result for later phases."""

    id: str = Field(default_factory=lambda: f"life-reflection-{uuid4().hex}")
    summary: str
    learned_context: str = ""
    source_event_ids: tuple[str, ...] = ()
    source_job_ids: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    memory_candidates: tuple[str, ...] = ()
    open_loop_updates: tuple[str, ...] = ()
    next_attention_items: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def default_attention_decay(now: datetime) -> datetime:
    """Return the default MVP attention decay timestamp."""

    return now + timedelta(hours=6)
