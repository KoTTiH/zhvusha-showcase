"""Desktop Control Skill Pack contracts."""

from __future__ import annotations


def test_desktop_control_policy_maps_legacy_actions_to_canonical_capabilities() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionKind,
        DesktopActionRequest,
        DesktopControlPolicy,
    )

    plan = DesktopControlPolicy().plan(
        DesktopActionRequest(
            action=DesktopActionKind.MEDIA_CONTROL,
            operation="pause",
            source="voice",
        )
    )

    assert plan.allowed is True
    assert plan.capability == "desktop_media_control"
    assert plan.requires_approval is True
    assert plan.risk == "low"
    assert plan.audit_event["source"] == "voice"
    assert plan.audit_event["dialogue_owner"] == "zhvusha"


def test_desktop_control_policy_rejects_shell_as_not_pack_member() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionKind,
        DesktopActionRequest,
        DesktopControlPolicy,
    )

    plan = DesktopControlPolicy().plan(
        DesktopActionRequest(
            action=DesktopActionKind.SHELL,
            operation="run",
            target="rm -rf /",
            source="voice",
        )
    )

    assert plan.allowed is False
    assert plan.capability == "desktop.shell"
    assert plan.requires_approval is True
    assert "not part of Desktop Control Skill Pack" in plan.reason


def test_desktop_power_actions_are_high_risk_and_approval_gated() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionKind,
        DesktopActionRequest,
        DesktopControlPolicy,
    )

    plan = DesktopControlPolicy().plan(
        DesktopActionRequest(
            action=DesktopActionKind.SYSTEM_POWER,
            operation="sleep",
            source="text",
        )
    )

    assert plan.allowed is True
    assert plan.capability == "desktop.system_power"
    assert plan.risk == "high"
    assert plan.requires_approval is True
