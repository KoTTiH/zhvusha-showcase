"""Pre-send gate for autonomous social actions."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.agency.judgement import SocialJudgement
from src.agency.models import (
    AgencyAuditEvent,
    SocialJudgementDecision,
    SocialJudgementInput,
    SocialPermissionGrant,
    SocialPermissionScope,
)
from src.agency.store import FileSocialPermissionStore  # noqa: TC001


class SocialSendRequest(BaseModel):
    """Candidate autonomous social send before any Telegram/tool execution."""

    target_id: str
    message: str
    judgement: SocialJudgementInput
    required_scope: SocialPermissionScope = SocialPermissionScope.REPLY_IF_ADDRESSED
    topic: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class SocialSendGateResult(BaseModel):
    """Structured pre-send result. This object never executes the send."""

    allowed: bool
    reason: str
    target_id: str
    grant_id: str = ""
    judgement: SocialJudgementDecision | None = None
    audit_event: AgencyAuditEvent


class SocialSendGate:
    """Check grant, scope, topic, rate and judgement before autonomous send."""

    def __init__(
        self,
        *,
        store: FileSocialPermissionStore,
        judgement: SocialJudgement | None = None,
    ) -> None:
        self._store = store
        self._judgement = judgement or SocialJudgement()

    def evaluate(
        self,
        request: SocialSendRequest,
        *,
        now: datetime | None = None,
    ) -> SocialSendGateResult:
        """Evaluate a social send candidate without granting tool approval."""
        if self._store.emergency_stop_enabled():
            return self._blocked(request, "emergency_stop")

        grant = self._matching_grant(
            target_id=request.target_id,
            scope=request.required_scope,
            now=now,
        )
        if grant is None:
            return self._blocked(request, "missing_active_grant")

        topic_reason = _topic_block_reason(grant=grant, topic=request.topic)
        if topic_reason:
            return self._blocked(request, topic_reason, grant=grant)

        judgement_input = request.judgement.model_copy(
            update={
                "target_id": request.target_id,
                "topic": request.topic or request.judgement.topic,
                "recent_messages_sent": max(
                    request.judgement.recent_messages_sent,
                    self._store.count_sent_in_window(
                        grant_id=grant.id,
                        now=now or datetime.now(tz=UTC),
                        window_seconds=grant.window_seconds,
                    ),
                ),
            }
        )
        judgement = self._judgement.evaluate(judgement_input, grant=grant, now=now)
        if not judgement.can_send:
            return self._blocked(
                request,
                f"social_judgement_{judgement.action.value}",
                grant=grant,
                judgement=judgement,
            )

        return SocialSendGateResult(
            allowed=True,
            reason="allowed_by_grant_and_judgement",
            target_id=request.target_id,
            grant_id=grant.id,
            judgement=judgement,
            audit_event=AgencyAuditEvent(
                event_type="social_send_allowed",
                reason=judgement.reason,
                target_id=request.target_id,
                grant_id=grant.id,
            ),
        )

    def _matching_grant(
        self,
        *,
        target_id: str,
        scope: SocialPermissionScope,
        now: datetime | None,
    ) -> SocialPermissionGrant | None:
        active = (
            grant
            for grant in self._store.list_grants()
            if grant.target_id == target_id and grant.permits(scope, now=now)
        )
        return next(active, None)

    @staticmethod
    def _blocked(
        request: SocialSendRequest,
        reason: str,
        *,
        grant: SocialPermissionGrant | None = None,
        judgement: SocialJudgementDecision | None = None,
    ) -> SocialSendGateResult:
        return SocialSendGateResult(
            allowed=False,
            reason=reason,
            target_id=request.target_id,
            grant_id=grant.id if grant is not None else "",
            judgement=judgement,
            audit_event=AgencyAuditEvent(
                event_type="social_send_blocked",
                reason=reason,
                target_id=request.target_id,
                grant_id=grant.id if grant is not None else "",
            ),
        )


def _topic_block_reason(*, grant: SocialPermissionGrant, topic: str) -> str:
    normalized = topic.strip().lower()
    if not normalized:
        return ""
    forbidden = {item.strip().lower() for item in grant.forbidden_topics}
    if normalized in forbidden:
        return "topic_forbidden_by_grant"
    allowed = {item.strip().lower() for item in grant.allowed_topics}
    if allowed and normalized not in allowed:
        return "topic_not_allowed_by_grant"
    return ""
