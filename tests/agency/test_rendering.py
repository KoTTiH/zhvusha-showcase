from __future__ import annotations

from src.agency.models import (
    AgencyAction,
    AgencyActionKind,
    AgencyDataNeed,
    AgencyIntent,
    AgencyIntentKind,
    AgencyOutcomeKind,
    AgencyPermissionRequest,
    AutonomyDecisionType,
    SocialPermissionScope,
    SocialTargetType,
)


def _intent() -> AgencyIntent:
    return AgencyIntent(
        id="agency-test",
        kind=AgencyIntentKind.SELF_COMPLEXIFICATION,
        source="personality_driven_agency",
        goal="Проверить social grants перед автономным сообщением",
        why_complexification="Жвуше нужен безопасный способ действовать через людей.",
        why_personality_matters="curiosity + social calibration",
        priority=82,
        drive_vector={"curiosity": 0.9, "care": 0.7},
        personality_drivers=("affect:curiosity", "feedback:correction"),
        safety_constraints=("ToolGateway gates side effects",),
        data_needs=(AgencyDataNeed.HUMAN_OPINION,),
        expected_outcomes=(AgencyOutcomeKind.SPEC, AgencyOutcomeKind.MEMORY_CANDIDATE),
        evidence=("memory/feedback.md:12",),
        candidate_actions=(
            AgencyAction(
                kind=AgencyActionKind.TELEGRAM_MCP_READ,
                capability="telegram_mcp_read",
                target_id="@devchat",
                description="Почитать чат как evidence.",
            ),
            AgencyAction(
                kind=AgencyActionKind.TELEGRAM_MCP_SEND,
                capability="telegram_mcp_send",
                target_id="@devchat",
                description="Спросить людей о живости Жвуши.",
                side_effect=True,
                permission_scope=SocialPermissionScope.REPLY_IF_ADDRESSED,
            ),
        ),
    )


def test_render_agency_intent_status_shows_personality_and_tool_reasons() -> None:
    from src.agency.rendering import render_agency_intent_status

    status = render_agency_intent_status(_intent())

    assert "AgencyIntent agency-test" in status
    assert "kind: self_complexification" in status
    assert "priority: 82" in status
    assert "goal: Проверить social grants" in status
    assert "why_personality_matters: curiosity + social calibration" in status
    assert "drive_vector: care=0.70, curiosity=0.90" in status
    assert "personality_drivers: affect:curiosity, feedback:correction" in status
    assert "safety_constraints: ToolGateway gates side effects" in status
    assert "data_needs: human_opinion" in status
    assert "expected_outcomes: spec, memory_candidate" in status
    assert "allowed_by_default: telegram_mcp_read -> @devchat" in status
    assert "side_effect_candidate: telegram_mcp_send -> @devchat" in status


def test_render_policy_decision_status_shows_blocks_and_permission_request() -> None:
    from src.agency.models import AgencyAuditEvent, AutonomyPolicyDecision
    from src.agency.rendering import render_policy_decision_status

    intent = _intent()
    decision = AutonomyPolicyDecision(
        decision=AutonomyDecisionType.ASK_NIKITA,
        reason="Social side effect requires scoped permission grant.",
        allowed_actions=(intent.candidate_actions[0],),
        blocked_actions=(intent.candidate_actions[1],),
        permission_request=AgencyPermissionRequest(
            target_id="@devchat",
            target_type=SocialTargetType.GROUP,
            requested_scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
            reason="Нужно спросить людей, но без grant нельзя писать.",
            duration_hint="1h",
        ),
        audit_event=AgencyAuditEvent(
            event_type="agency_permission_required",
            reason="Missing social permission grant",
            intent_id=intent.id,
            target_id="@devchat",
        ),
    )

    status = render_policy_decision_status(decision)

    assert "AutonomyPolicy: ask_nikita" in status
    assert "reason: Social side effect requires scoped permission grant." in status
    assert "allowed: telegram_mcp_read -> @devchat" in status
    assert "blocked: telegram_mcp_send -> @devchat" in status
    assert "permission_request: group @devchat" in status
    assert "scopes: reply_if_addressed" in status
    assert "duration_hint: 1h" in status
    assert "audit_event: agency_permission_required" in status
