"""Autonomy policy for agency actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.agency.models import (
    AgencyAction,
    AgencyAuditEvent,
    AgencyIntent,
    AgencyPermissionRequest,
    AutonomyDecisionType,
    AutonomyPolicyDecision,
    SocialPermissionGrant,
    SocialPermissionScope,
)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class _PolicyBuckets:
    allowed: list[AgencyAction]
    blocked: list[AgencyAction]
    tier3_actions: list[AgencyAction]
    permission_request: AgencyPermissionRequest | None = None
    used_grant: SocialPermissionGrant | None = None


class AutonomyPolicy:
    """Separate autonomous read/draft work from gated side effects."""

    def __init__(self, *, emergency_stop: bool = False) -> None:
        self._emergency_stop = emergency_stop

    def decide(
        self,
        intent: AgencyIntent,
        *,
        grants: tuple[SocialPermissionGrant, ...] = (),
        now: datetime | None = None,
    ) -> AutonomyPolicyDecision:
        """Evaluate which candidate actions may proceed."""

        buckets = _PolicyBuckets(allowed=[], blocked=[], tier3_actions=[])
        for action in intent.candidate_actions:
            _evaluate_action(
                action,
                buckets=buckets,
                grants=grants,
                now=now,
                emergency_stop=self._emergency_stop,
            )

        if self._emergency_stop and buckets.blocked:
            return AutonomyPolicyDecision(
                decision=AutonomyDecisionType.BLOCKED,
                reason="Emergency stop blocks autonomous side-effect actions.",
                allowed_actions=tuple(buckets.allowed),
                blocked_actions=tuple(buckets.blocked),
                audit_event=AgencyAuditEvent(
                    event_type="agency_policy_blocked",
                    reason="Emergency stop active",
                    intent_id=intent.id,
                    target_id=_first_target(buckets.blocked),
                ),
            )
        if buckets.permission_request is not None:
            return AutonomyPolicyDecision(
                decision=AutonomyDecisionType.ASK_NIKITA,
                reason="Social side effect requires scoped permission grant.",
                allowed_actions=tuple(buckets.allowed),
                blocked_actions=tuple(buckets.blocked),
                permission_request=buckets.permission_request,
                audit_event=AgencyAuditEvent(
                    event_type="agency_permission_required",
                    reason="Missing social permission grant",
                    intent_id=intent.id,
                    target_id=buckets.permission_request.target_id,
                ),
            )
        if buckets.tier3_actions:
            return AutonomyPolicyDecision(
                decision=AutonomyDecisionType.ASK_NIKITA,
                reason="Tier 3 self-change requires Никита approval.",
                allowed_actions=tuple(buckets.allowed),
                blocked_actions=tuple(buckets.blocked),
                audit_event=AgencyAuditEvent(
                    event_type="agency_tier3_nikita_required",
                    reason="Tier 3 cannot be self-approved by Жвуша",
                    intent_id=intent.id,
                    target_id=_first_target(buckets.tier3_actions),
                ),
            )
        if buckets.blocked:
            return AutonomyPolicyDecision(
                decision=AutonomyDecisionType.APPROVAL_REQUIRED,
                reason="Side-effect action requires explicit approval.",
                allowed_actions=tuple(buckets.allowed),
                blocked_actions=tuple(buckets.blocked),
                audit_event=AgencyAuditEvent(
                    event_type="agency_approval_required",
                    reason="Side-effect action has no autonomous policy grant",
                    intent_id=intent.id,
                    target_id=_first_target(buckets.blocked),
                ),
            )
        return AutonomyPolicyDecision(
            decision=AutonomyDecisionType.AUTO,
            reason="Allowed by autonomy policy.",
            allowed_actions=tuple(buckets.allowed),
            audit_event=AgencyAuditEvent(
                event_type="agency_policy_allowed",
                reason="Allowed by policy and grants",
                intent_id=intent.id,
                grant_id=buckets.used_grant.id
                if buckets.used_grant is not None
                else "",
            ),
        )


def _evaluate_action(
    action: AgencyAction,
    *,
    buckets: _PolicyBuckets,
    grants: tuple[SocialPermissionGrant, ...],
    now: datetime | None,
    emergency_stop: bool,
) -> None:
    if action.risk_tier >= 3:
        buckets.tier3_actions.append(action)
        (buckets.blocked if action.side_effect else buckets.allowed).append(action)
        return
    if not action.side_effect:
        buckets.allowed.append(action)
        return
    if emergency_stop:
        buckets.blocked.append(action)
        return
    required_scope = action.permission_scope
    if required_scope is None:
        buckets.blocked.append(action)
        return
    grant = _find_grant(
        grants,
        target_id=action.target_id,
        scope=required_scope,
        now=now,
    )
    if grant is None:
        buckets.blocked.append(action)
        buckets.permission_request = _permission_request(action, required_scope)
        return
    buckets.used_grant = grant
    buckets.allowed.append(
        action.model_copy(update={"requires_social_judgement": True})
    )


def _permission_request(
    action: AgencyAction,
    required_scope: SocialPermissionScope,
) -> AgencyPermissionRequest:
    return AgencyPermissionRequest(
        target_id=action.target_id,
        requested_scopes=(required_scope,),
        reason=action.description or "Жвуша хочет выполнить social action.",
    )


def _find_grant(
    grants: tuple[SocialPermissionGrant, ...],
    *,
    target_id: str,
    scope: SocialPermissionScope,
    now: datetime | None,
) -> SocialPermissionGrant | None:
    for grant in grants:
        if grant.target_id == target_id and grant.permits(scope, now=now):
            return grant
    return None


def _first_target(actions: list[AgencyAction]) -> str:
    for action in actions:
        if action.target_id:
            return action.target_id
    return ""
