"""Safe status rendering for Agency intents and policy decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agency.models import (
        AgencyAction,
        AgencyIntent,
        AutonomyPolicyDecision,
    )


def render_agency_intent_status(intent: AgencyIntent) -> str:
    """Render operator-facing AgencyIntent status without executing actions."""

    lines = [
        f"AgencyIntent {intent.id}",
        f"kind: {intent.kind.value}",
        f"source: {intent.source}",
        f"priority: {intent.priority}",
        f"goal: {intent.goal}",
        f"why_complexification: {intent.why_complexification}",
    ]
    _append_line(lines, "why_personality_matters", intent.why_personality_matters)
    if intent.drive_vector:
        lines.append(f"drive_vector: {_render_drive_vector(intent.drive_vector)}")
    _append_sequence(lines, "personality_drivers", intent.personality_drivers)
    _append_sequence(lines, "safety_constraints", intent.safety_constraints)
    _append_sequence(
        lines, "data_needs", tuple(item.value for item in intent.data_needs)
    )
    _append_sequence(
        lines,
        "expected_outcomes",
        tuple(item.value for item in intent.expected_outcomes),
    )
    if intent.candidate_actions:
        lines.append("candidate_actions:")
        for action in intent.candidate_actions:
            prefix = (
                "side_effect_candidate" if action.side_effect else "allowed_by_default"
            )
            lines.append(f"- {prefix}: {_render_action(action)}")
    if intent.evidence:
        lines.append(f"evidence_count: {len(intent.evidence)}")
    return "\n".join(lines)


def render_policy_decision_status(decision: AutonomyPolicyDecision) -> str:
    """Render a policy decision for Жвушин orchestrator/status surfaces."""

    lines = [
        f"AutonomyPolicy: {decision.decision.value}",
        f"reason: {decision.reason}",
    ]
    if decision.allowed_actions:
        lines.append("allowed_actions:")
        lines.extend(
            f"- allowed: {_render_action(action)}"
            for action in decision.allowed_actions
        )
    if decision.blocked_actions:
        lines.append("blocked_actions:")
        lines.extend(
            f"- blocked: {_render_action(action)}"
            for action in decision.blocked_actions
        )
    if decision.permission_request is not None:
        request = decision.permission_request
        lines.extend(
            (
                f"permission_request: {request.target_type.value} {request.target_id}",
                "scopes: "
                + ", ".join(scope.value for scope in request.requested_scopes),
                f"permission_reason: {request.reason}",
            )
        )
        _append_line(lines, "duration_hint", request.duration_hint)
    lines.append(f"audit_event: {decision.audit_event.event_type}")
    _append_line(lines, "audit_reason", decision.audit_event.reason)
    return "\n".join(lines)


def _render_action(action: AgencyAction) -> str:
    target = f" -> {action.target_id}" if action.target_id else ""
    description = f" — {action.description}" if action.description else ""
    risk = f" risk_tier={action.risk_tier}"
    capability = f" capability={action.capability}" if action.capability else ""
    return f"{action.kind.value}{target}{capability}{risk}{description}"


def _render_drive_vector(drive_vector: dict[str, float]) -> str:
    return ", ".join(
        f"{key}={max(0.0, min(float(value), 1.0)):.2f}"
        for key, value in sorted(drive_vector.items())
    )


def _append_sequence(lines: list[str], label: str, values: tuple[str, ...]) -> None:
    if values:
        lines.append(f"{label}: {', '.join(values)}")


def _append_line(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")
