"""Value models for the shared Agent Runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentJobStatus(StrEnum):
    """Durable job lifecycle states."""

    DRAFT = "draft"
    AWAITING_INPUT = "awaiting_input"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"
    NEEDS_REVIEW = "needs_review"


class FindingStatus(StrEnum):
    """Evidence status for a finding inside a Context Capsule."""

    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    UNCONFIRMED = "unconfirmed"
    REJECTED = "rejected"


class AgentEventType(StrEnum):
    """Runtime event kinds stored for audit and Telegram rendering."""

    CREATED = "created"
    STARTED = "started"
    PROGRESS = "progress"
    FOLLOWUP_ATTACHED = "followup_attached"
    ARTIFACT_ATTACHED = "artifact_attached"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    RECOVERED = "recovered"
    MEMORY_STAGED = "memory_staged"


class Finding(BaseModel):
    """Single evidence-backed claim produced by an agent."""

    claim: str
    status: FindingStatus
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()


class ContextCapsule(BaseModel):
    """Structured result that a worker returns to Zhvusha."""

    summary: str
    processed_context: str = ""
    findings: tuple[Finding, ...] = ()
    sources: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    memory_candidates: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    markdown_report: str = ""


class ContextPack(BaseModel):
    """Input context prepared by Zhvusha for one agent invocation."""

    user_request: str
    chat_context: tuple[str, ...] = ()
    active_code_state: str = ""
    attachments: tuple[str, ...] = ()
    relevant_files: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


class AgentDefinition(BaseModel):
    """Reusable agent profile metadata."""

    id: str
    version: int = 1
    purpose: str
    default_worker: str
    allowed_capabilities: tuple[str, ...] = ()
    output_schema: str = "context_capsule.v1"
    safety_policy: str = "readonly.v1"


class CapabilityDefinition(BaseModel):
    """Runtime capability exposed through the Tool Gateway."""

    id: str
    description: str = ""
    risk: str = "low"
    requires_approval: bool = False


class InvocationProfile(BaseModel):
    """Concrete capabilities granted to a single worker run."""

    id: str
    worker: str = "codex_cli"
    allowed_capabilities: tuple[str, ...] = ()
    denied_capabilities: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    def allows(self, capability: str) -> bool:
        """Return whether a capability is physically available in this run."""
        if capability in set(self.denied_capabilities):
            return False
        return capability in set(self.allowed_capabilities)


class AgentEvent(BaseModel):
    """Machine-readable event emitted by the runtime."""

    job_id: str
    event_type: AgentEventType
    message: str = ""
    payload: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentJob(BaseModel):
    """Durable job record."""

    id: str
    owner_user_id: int
    chat_id: int
    source_message_id: str
    fingerprint: str
    kind: str
    profile: InvocationProfile
    context_pack: ContextPack
    status: AgentJobStatus = AgentJobStatus.QUEUED
    followups: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    result: ContextCapsule | None = None
    error: str = ""
    observability: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def new(
        cls,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        fingerprint: str,
        kind: str,
        profile: InvocationProfile,
        context_pack: ContextPack,
        status: AgentJobStatus = AgentJobStatus.QUEUED,
    ) -> AgentJob:
        """Create a new job with a generated id."""
        return cls(
            id=f"job-{uuid4().hex}",
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            fingerprint=fingerprint,
            kind=kind,
            profile=profile,
            context_pack=context_pack,
            status=status,
        )

    def with_status(
        self,
        status: AgentJobStatus,
        *,
        error: str = "",
        result: ContextCapsule | None = None,
    ) -> AgentJob:
        """Return a copy with lifecycle timestamps updated consistently."""
        now = datetime.now(UTC)
        changes: dict[str, object] = {
            "status": status,
            "updated_at": now,
        }
        if status is AgentJobStatus.RUNNING and self.started_at is None:
            changes["started_at"] = now
        if status in {
            AgentJobStatus.DONE,
            AgentJobStatus.FAILED,
            AgentJobStatus.CANCELED,
            AgentJobStatus.NEEDS_REVIEW,
        }:
            changes["finished_at"] = now
        if error:
            changes["error"] = error
        if result is not None:
            changes["result"] = result
        return self.model_copy(update=changes)
