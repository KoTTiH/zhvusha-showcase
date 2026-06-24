"""Daemon-side planner for bounded Agent Runtime job requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from src.agent_runtime.models import (
    AgentJob,
    AgentJobStatus,
    ContextPack,
    InvocationProfile,
)

if TYPE_CHECKING:
    from src.agent_runtime.topic_signals import TopicClusterReadySignal

READONLY_DAEMON_PROFILE_IDS: tuple[str, ...] = (
    "source_compare.readonly",
    "self_coding.readonly_discussion",
    "agency.readonly_draft",
    "web_research.readonly",
    "channel_visual.readonly_artifacts",
    "telegram_mcp.personal_readonly",
)

SIDE_EFFECT_CAPABILITIES: frozenset[str] = frozenset(
    {
        "browser_submit",
        "commit",
        "commit_after_gate",
        "edit_env",
        "publish",
        "restart",
        "send_message",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
        "telegram_mcp_modify",
        "telegram_mcp_send",
        "write_files",
        "write_whitelisted_files_after_approval",
    }
)

_TOPIC_ROUTE_PROFILES: dict[str, str] = {
    "spec": "self_coding.readonly_discussion",
    "proposal": "agency.readonly_draft",
    "post": "agency.readonly_draft",
    "report": "agency.readonly_draft",
}


class DaemonAgentRuntimeJobRequest(BaseModel):
    """A daemon signal converted into a candidate Agent Runtime job."""

    kind: str
    profile_id: str
    user_request: str
    source_signal_id: str = ""
    chat_id: int = 0
    owner_user_id: int = 0
    source_message_id: str = ""
    fingerprint: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class DaemonAgentRuntimeJobPlan(BaseModel):
    """Preflight result for a daemon-created Agent Runtime request."""

    allowed: bool
    reason: str
    request: DaemonAgentRuntimeJobRequest
    profile: InvocationProfile | None = None
    context_pack: ContextPack | None = None
    can_enqueue: bool = False


class DaemonAgentRuntimeEnqueueResult(BaseModel):
    """Result of a daemon-created Agent Runtime enqueue attempt."""

    allowed: bool
    reason: str
    plan: DaemonAgentRuntimeJobPlan
    job_id: str = ""
    job_status: AgentJobStatus | None = None


class AgentRuntimeJobCreator(Protocol):
    """Narrow Agent Runtime interface used by daemon requesters/tools."""

    async def create_job(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        fingerprint: str,
        kind: str,
        profile: InvocationProfile,
        context_pack: ContextPack,
        status: AgentJobStatus = AgentJobStatus.QUEUED,
    ) -> AgentJob: ...


class DaemonAgentRuntimeJobRequester:
    """Build safe Agent Runtime job plans from daemon signals."""

    def __init__(
        self,
        *,
        profiles: Mapping[str, InvocationProfile],
        enabled: bool = False,
        allowed_profile_ids: Iterable[str] = READONLY_DAEMON_PROFILE_IDS,
    ) -> None:
        self._profiles = dict(profiles)
        self._enabled = enabled
        self._allowed_profile_ids = frozenset(allowed_profile_ids)

    @classmethod
    def from_profiles(
        cls,
        profiles: Iterable[InvocationProfile] | Mapping[str, InvocationProfile],
        *,
        enabled: bool = False,
        allowed_profile_ids: Iterable[str] = READONLY_DAEMON_PROFILE_IDS,
    ) -> DaemonAgentRuntimeJobRequester:
        """Create a requester from built-in or injected invocation profiles."""
        if isinstance(profiles, Mapping):
            profiles_by_id = dict(profiles)
        else:
            profiles_by_id = {profile.id: profile for profile in profiles}
        return cls(
            profiles=profiles_by_id,
            enabled=enabled,
            allowed_profile_ids=allowed_profile_ids,
        )

    def plan(self, request: DaemonAgentRuntimeJobRequest) -> DaemonAgentRuntimeJobPlan:
        """Return a safe preflight plan without enqueueing or executing a job."""
        if not self._enabled:
            return self._blocked(request, "requester_disabled")

        profile = self._profiles.get(request.profile_id)
        if profile is None:
            return self._blocked(request, "unknown_profile")

        if profile.id not in self._allowed_profile_ids:
            return self._blocked(request, "profile_not_allowed_for_daemon", profile)

        denied = sorted(
            SIDE_EFFECT_CAPABILITIES.intersection(profile.allowed_capabilities)
        )
        if denied:
            return self._blocked(
                request,
                f"side_effect_capability_denied:{','.join(denied)}",
                profile,
            )

        metadata = dict(request.metadata)
        if request.source_signal_id:
            metadata["daemon_source_signal_id"] = request.source_signal_id
        if request.fingerprint:
            metadata["daemon_fingerprint"] = request.fingerprint
        if request.source_message_id:
            metadata["daemon_source_message_id"] = request.source_message_id

        return DaemonAgentRuntimeJobPlan(
            allowed=True,
            reason="ready",
            request=request,
            profile=profile,
            context_pack=ContextPack(
                user_request=request.user_request,
                constraints=(
                    "daemon_requester_readonly_preflight",
                    "side_effects_require_separate_approval",
                ),
                metadata=metadata,
            ),
            can_enqueue=True,
        )

    async def enqueue(
        self,
        request: DaemonAgentRuntimeJobRequest,
        runtime: AgentRuntimeJobCreator,
    ) -> DaemonAgentRuntimeEnqueueResult:
        """Create a queued Agent Runtime job after daemon-side preflight."""

        plan = self.plan(request)
        if not plan.allowed or not plan.can_enqueue:
            return DaemonAgentRuntimeEnqueueResult(
                allowed=False,
                reason=plan.reason,
                plan=plan,
            )
        if plan.profile is None or plan.context_pack is None:
            return DaemonAgentRuntimeEnqueueResult(
                allowed=False,
                reason="invalid_plan",
                plan=plan,
            )

        try:
            job = await runtime.create_job(
                owner_user_id=request.owner_user_id,
                chat_id=request.chat_id,
                source_message_id=request.source_message_id or request.source_signal_id,
                fingerprint=_job_fingerprint(request),
                kind=request.kind,
                profile=plan.profile,
                context_pack=plan.context_pack,
                status=AgentJobStatus.QUEUED,
            )
        except Exception as exc:
            return DaemonAgentRuntimeEnqueueResult(
                allowed=False,
                reason=f"runtime_create_failed:{type(exc).__name__}",
                plan=plan,
            )

        return DaemonAgentRuntimeEnqueueResult(
            allowed=True,
            reason="queued",
            plan=plan,
            job_id=job.id,
            job_status=job.status,
        )

    @staticmethod
    def _blocked(
        request: DaemonAgentRuntimeJobRequest,
        reason: str,
        profile: InvocationProfile | None = None,
    ) -> DaemonAgentRuntimeJobPlan:
        return DaemonAgentRuntimeJobPlan(
            allowed=False,
            reason=reason,
            request=request,
            profile=profile,
        )


def render_daemon_agent_runtime_plan_status(
    plan: DaemonAgentRuntimeJobPlan,
) -> str:
    """Render daemon Agent Runtime preflight status without enqueueing jobs."""

    profile_id = plan.profile.id if plan.profile is not None else "missing"
    lines = [
        f"Daemon Agent Runtime request: {plan.reason}",
        f"allowed: {_yes_no(plan.allowed)}",
        f"can_enqueue: {_yes_no(plan.can_enqueue)}",
        f"kind: {plan.request.kind}",
        f"profile: {profile_id}",
    ]
    if plan.request.source_signal_id:
        lines.append(f"source_signal_id: {plan.request.source_signal_id}")
    if plan.request.source_message_id:
        lines.append(f"source_message_id: {plan.request.source_message_id}")
    if plan.request.fingerprint:
        lines.append("fingerprint: present")
    if plan.context_pack is not None:
        lines.append(f"constraints: {', '.join(plan.context_pack.constraints)}")
    lines.append("execution: not_started")
    return "\n".join(lines)


def render_daemon_agent_runtime_enqueue_status(
    result: DaemonAgentRuntimeEnqueueResult,
) -> str:
    """Render daemon Agent Runtime enqueue status without raw user request text."""

    lines = [
        f"Daemon Agent Runtime enqueue: {result.reason}",
        f"allowed: {_yes_no(result.allowed)}",
        f"kind: {result.plan.request.kind}",
        f"profile: {result.plan.request.profile_id}",
    ]
    if result.job_id:
        lines.append(f"job_id: {result.job_id}")
    if result.job_status is not None:
        lines.append(f"job_status: {result.job_status.value}")
    if result.plan.request.source_signal_id:
        lines.append(f"source_signal_id: {result.plan.request.source_signal_id}")
    lines.append("execution: queued_only")
    return "\n".join(lines)


def build_daemon_request_from_topic_signal(
    signal: TopicClusterReadySignal,
    *,
    owner_user_id: int = 0,
    chat_id: int = 0,
) -> DaemonAgentRuntimeJobRequest:
    """Convert a topic backlog signal into a bounded daemon runtime request."""

    route = signal.recommended_route
    profile_id = _TOPIC_ROUTE_PROFILES[route]
    source_url = signal.payload.get("source_url_0", "")
    evidence = f"\nEvidence: {source_url}" if source_url else ""
    return DaemonAgentRuntimeJobRequest(
        kind=f"topic_signal.{route}",
        profile_id=profile_id,
        user_request=(
            f"Prepare a bounded {route} candidate from topic signal "
            f"{signal.cluster_key}.\n"
            f"Title: {signal.title}\n"
            f"Summary: {signal.summary}\n"
            f"Tier: {signal.tier}\n"
            "Do not publish, send, write files, or execute implementation. "
            "Return a Context Capsule proposal for Жвуша."
            f"{evidence}"
        ),
        source_signal_id=f"topic_cluster_ready:{signal.cluster_key}",
        chat_id=chat_id,
        owner_user_id=owner_user_id,
        source_message_id=f"topic:{signal.cluster_key}",
        fingerprint=f"topic_signal:{signal.cluster_key}:{route}",
        metadata={
            "topic_cluster_key": signal.cluster_key,
            "topic_recommended_route": route,
            "topic_tier": str(signal.tier),
            "topic_requires_approval": _bool_text(signal.requires_approval),
            "topic_requires_nikita": _bool_text(signal.requires_nikita),
            "topic_auto_publish_allowed": _bool_text(signal.auto_publish_allowed),
            "topic_auto_execute_allowed": _bool_text(signal.auto_execute_allowed),
        },
    )


def _job_fingerprint(request: DaemonAgentRuntimeJobRequest) -> str:
    if request.fingerprint:
        return request.fingerprint
    stable_source = request.source_signal_id or request.source_message_id
    if not stable_source:
        stable_source = sha256(request.user_request.encode("utf-8")).hexdigest()[:16]
    return f"daemon:{request.kind}:{request.profile_id}:{stable_source}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
