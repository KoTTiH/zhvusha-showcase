"""Value models for Agency / Self-Complexification Runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class AgencyIntentKind(StrEnum):
    """Why this agency intent exists."""

    SELF_COMPLEXIFICATION = "self_complexification"
    SOCIAL_LEARNING = "social_learning"
    RESEARCH = "research"
    SELF_CODING = "self_coding"
    MEMORY_REFLECTION = "memory_reflection"


class AgencyDataNeed(StrEnum):
    """Type of missing data the intent needs."""

    FACTS = "facts"
    HUMAN_OPINION = "human_opinion"
    OBSERVATION = "observation"
    CODE = "code"
    MEMORY = "memory"
    HISTORY = "history"


class AgencyOutcomeKind(StrEnum):
    """Expected artifact or next step from an agency pass."""

    CONTEXT_CAPSULE = "context_capsule"
    MEMORY_CANDIDATE = "memory_candidate"
    DRAFT = "draft"
    SPEC = "spec"
    CODE_TASK = "code_task"
    ASK_NIKITA = "ask_nikita"
    SOCIAL_ACTION = "social_action"


class AgencyActionKind(StrEnum):
    """Concrete candidate action class."""

    READ_WORKSPACE = "read_workspace"
    SEARCH_KB = "search_kb"
    WEB_RESEARCH = "web_research"
    TELEGRAM_MCP_READ = "telegram_mcp_read"
    TELEGRAM_MCP_SEND = "telegram_mcp_send"
    DRAFT_MESSAGE = "draft_message"
    CREATE_SPEC = "create_spec"
    STAGE_MEMORY = "stage_memory"


class SocialPermissionScope(StrEnum):
    """Scoped permission grant for a person/chat/group/forum surface."""

    READ_ONLY = "read_only"
    MONITOR = "monitor"
    REPLY_IF_ADDRESSED = "reply_if_addressed"
    INITIATE_ONCE = "initiate_once"
    INITIATE_OCCASIONALLY = "initiate_occasionally"
    FREE_CONVERSATION = "free_conversation"
    JOIN_GROUP = "join_group"
    JOIN_AND_OBSERVE = "join_and_observe"
    DISCUSS_TOPIC = "discuss_topic"
    TOPIC_LIMITED_POSTING = "topic_limited_posting"


class SocialTargetType(StrEnum):
    """Social surface type."""

    PERSON = "person"
    CHAT = "chat"
    GROUP = "group"
    CHANNEL = "channel"
    FORUM = "forum"


class SocialPermissionStatus(StrEnum):
    """Lifecycle state for a social grant."""

    ACTIVE = "active"
    PAUSED = "paused"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AutonomyDecisionType(StrEnum):
    """Policy-level outcome before execution."""

    AUTO = "auto"
    DRAFT = "draft"
    ASK_NIKITA = "ask_nikita"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED = "blocked"
    FORBIDDEN = "forbidden"


class SocialJudgementAction(StrEnum):
    """Interaction judgement result. Includes the right to stay silent."""

    IGNORE = "ignore"
    READ_ONLY = "read_only"
    WAIT = "wait"
    DRAFT = "draft"
    REPLY = "reply"
    SEND = "send"
    ASK_NIKITA = "ask_nikita"
    LEAVE_CHAT = "leave_chat"
    MUTE_TOPIC = "mute_topic"


class AgencyAction(BaseModel):
    """One candidate tool/action selected for an agency intent."""

    kind: AgencyActionKind
    capability: str = ""
    target_id: str = ""
    description: str = ""
    side_effect: bool = False
    risk_tier: int = Field(default=1, ge=1, le=3)
    permission_scope: SocialPermissionScope | None = None
    requires_social_judgement: bool = False

    @field_validator("capability", "target_id", "description")
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        return value.strip()


class AgencyIntent(BaseModel):
    """Durable unit of Жвуша's self-generated intent."""

    id: str = Field(default_factory=lambda: f"agency-{uuid4().hex}")
    kind: AgencyIntentKind
    source: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    why_complexification: str = Field(min_length=1)
    why_personality_matters: str = ""
    priority: int = Field(default=50, ge=0, le=100)
    drive_vector: dict[str, float] = Field(default_factory=dict)
    personality_drivers: tuple[str, ...] = ()
    safety_constraints: tuple[str, ...] = ()
    data_needs: tuple[AgencyDataNeed, ...] = ()
    candidate_actions: tuple[AgencyAction, ...] = ()
    expected_outcomes: tuple[AgencyOutcomeKind, ...] = ()
    evidence: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("source", "goal", "why_complexification")
    @classmethod
    def _strip_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-blank after stripping whitespace")
        return cleaned

    @field_validator("evidence")
    @classmethod
    def _strip_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("evidence entries must be non-empty")
        return cleaned

    @field_validator("personality_drivers", "safety_constraints")
    @classmethod
    def _strip_tuple_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("entries must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _requires_data_or_action(self) -> AgencyIntent:
        if not self.data_needs and not self.candidate_actions:
            raise ValueError("AgencyIntent requires data_needs or candidate_actions")
        return self


class AgencyPermissionRequest(BaseModel):
    """Structured ask to Никита for a scoped social permission."""

    target_id: str
    target_type: SocialTargetType = SocialTargetType.CHAT
    requested_scopes: tuple[SocialPermissionScope, ...]
    reason: str
    risk: str = "medium"
    duration_hint: str = ""


class AgencyAuditEvent(BaseModel):
    """Human-readable and machine-readable agency policy event."""

    event_type: str
    reason: str
    intent_id: str = ""
    target_id: str = ""
    grant_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AutonomyPolicyDecision(BaseModel):
    """Result of autonomy policy before any action executes."""

    decision: AutonomyDecisionType
    reason: str
    allowed_actions: tuple[AgencyAction, ...] = ()
    blocked_actions: tuple[AgencyAction, ...] = ()
    permission_request: AgencyPermissionRequest | None = None
    audit_event: AgencyAuditEvent


class SocialPermissionGrant(BaseModel):
    """Scoped permission for a social target."""

    id: str = Field(default_factory=lambda: f"grant-{uuid4().hex}")
    target_id: str
    target_type: SocialTargetType
    scopes: tuple[SocialPermissionScope, ...]
    status: SocialPermissionStatus = SocialPermissionStatus.ACTIVE
    max_messages_per_window: int = Field(default=3, ge=0)
    window_seconds: int = Field(default=3600, ge=1)
    allowed_topics: tuple[str, ...] = ()
    forbidden_topics: tuple[str, ...] = ()
    privacy_level: str = "normal"
    granted_by: str = "nikita"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    @field_validator(
        "target_id",
        "privacy_level",
        "granted_by",
    )
    @classmethod
    def _strip_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-blank after stripping whitespace")
        return cleaned

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the grant may currently be used."""
        current = now or datetime.now(UTC)
        if self.status is not SocialPermissionStatus.ACTIVE:
            return False
        return self.expires_at is None or self.expires_at > current

    def permits(
        self,
        scope: SocialPermissionScope,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return whether a scope is available under this grant."""
        return self.is_active(now=now) and scope in set(self.scopes)


class SocialJudgementInput(BaseModel):
    """Runtime facts used before speaking in a social surface."""

    target_id: str
    topic: str = ""
    addressed_to_zhvusha: bool = False
    has_value_to_add: bool = False
    recent_messages_sent: int = Field(default=0, ge=0)
    repeats_obvious: bool = False
    conflict_or_private: bool = False
    privacy_risk: bool = False
    tone_ok: bool = True


class SocialJudgementDecision(BaseModel):
    """Decision made by social judgement."""

    action: SocialJudgementAction
    reason: str
    can_send: bool = False
    grant_id: str = ""
