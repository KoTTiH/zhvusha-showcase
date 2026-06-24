from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agency.models import (
    AgencyAction,
    AgencyActionKind,
    AgencyDataNeed,
    AgencyIntent,
    AgencyIntentKind,
    AgencyOutcomeKind,
    AutonomyDecisionType,
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialTargetType,
)
from src.agency.policy import AutonomyPolicy


def _human_opinion_intent() -> AgencyIntent:
    return AgencyIntent(
        kind=AgencyIntentKind.SELF_COMPLEXIFICATION,
        source="personality/desire",
        goal="Понять, как люди воспринимают живость Жвуши в групповом чате",
        why_complexification="Нужен человеческий взгляд, не только factual research.",
        data_needs=(AgencyDataNeed.HUMAN_OPINION,),
        expected_outcomes=(
            AgencyOutcomeKind.CONTEXT_CAPSULE,
            AgencyOutcomeKind.MEMORY_CANDIDATE,
        ),
        evidence=("personality.genes.curiosity=high", "desire: social calibration"),
        candidate_actions=(
            AgencyAction(
                kind=AgencyActionKind.TELEGRAM_MCP_READ,
                capability="telegram_mcp_read",
                target_id="@devchat",
                description="Почитать разрешённый чат перед выводами.",
            ),
            AgencyAction(
                kind=AgencyActionKind.TELEGRAM_MCP_SEND,
                capability="telegram_mcp_send",
                target_id="@devchat",
                description="Спросить людей, как звучит Жвуша.",
                side_effect=True,
                permission_scope=SocialPermissionScope.REPLY_IF_ADDRESSED,
            ),
        ),
    )


def test_personality_desire_intent_selects_readonly_tool_and_blocks_social_send_without_grant() -> (
    None
):
    decision = AutonomyPolicy().decide(_human_opinion_intent(), grants=())

    assert decision.decision is AutonomyDecisionType.ASK_NIKITA
    assert [action.kind for action in decision.allowed_actions] == [
        AgencyActionKind.TELEGRAM_MCP_READ
    ]
    assert [action.kind for action in decision.blocked_actions] == [
        AgencyActionKind.TELEGRAM_MCP_SEND
    ]
    assert decision.permission_request is not None
    assert decision.permission_request.target_id == "@devchat"
    assert (
        SocialPermissionScope.REPLY_IF_ADDRESSED
        in decision.permission_request.requested_scopes
    )
    assert "grant" in decision.audit_event.reason.lower()


def test_social_send_inside_grant_is_allowed_but_still_requires_judgement() -> None:
    now = datetime(2026, 5, 13, tzinfo=UTC)
    grant = SocialPermissionGrant(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        expires_at=now + timedelta(hours=1),
        max_messages_per_window=3,
    )

    decision = AutonomyPolicy().decide(
        _human_opinion_intent(),
        grants=(grant,),
        now=now,
    )

    assert decision.decision is AutonomyDecisionType.AUTO
    assert [action.kind for action in decision.allowed_actions] == [
        AgencyActionKind.TELEGRAM_MCP_READ,
        AgencyActionKind.TELEGRAM_MCP_SEND,
    ]
    send_action = decision.allowed_actions[1]
    assert send_action.requires_social_judgement is True
    assert decision.audit_event.grant_id == grant.id


def test_emergency_stop_blocks_autonomous_side_effects_even_with_grant() -> None:
    now = datetime(2026, 5, 13, tzinfo=UTC)
    grant = SocialPermissionGrant(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        expires_at=now + timedelta(hours=1),
    )

    decision = AutonomyPolicy(emergency_stop=True).decide(
        _human_opinion_intent(),
        grants=(grant,),
        now=now,
    )

    assert decision.decision is AutonomyDecisionType.BLOCKED
    assert [action.kind for action in decision.allowed_actions] == [
        AgencyActionKind.TELEGRAM_MCP_READ
    ]
    assert [action.kind for action in decision.blocked_actions] == [
        AgencyActionKind.TELEGRAM_MCP_SEND
    ]
    assert "emergency" in decision.reason.lower()


def test_tier3_self_change_spec_requires_nikita_not_auto_approval() -> None:
    intent = AgencyIntent(
        kind=AgencyIntentKind.SELF_COMPLEXIFICATION,
        source="personality/desire",
        goal="Изменить personality/runtime contract Жвуши",
        why_complexification="Жвуша хочет усложнить свою архитектуру.",
        data_needs=(AgencyDataNeed.CODE,),
        expected_outcomes=(AgencyOutcomeKind.SPEC,),
        candidate_actions=(
            AgencyAction(
                kind=AgencyActionKind.CREATE_SPEC,
                capability="request_tier3_specs_for_nikita_approval",
                description="Подготовить Tier 3 spec для Никиты.",
                risk_tier=3,
            ),
        ),
    )

    decision = AutonomyPolicy().decide(intent)

    assert decision.decision is AutonomyDecisionType.ASK_NIKITA
    assert [action.kind for action in decision.allowed_actions] == [
        AgencyActionKind.CREATE_SPEC
    ]
    assert decision.blocked_actions == ()
    assert decision.permission_request is None
    assert decision.audit_event.event_type == "agency_tier3_nikita_required"
