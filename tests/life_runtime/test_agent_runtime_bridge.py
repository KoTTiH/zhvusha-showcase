"""LifeRuntime Agent Runtime bridge contract."""

from __future__ import annotations


def test_life_reflection_action_request_is_read_only() -> None:
    from src.life_runtime import (
        LIFE_RUNTIME_READONLY_DENIED_CAPABILITIES,
        build_life_reflection_action_request,
    )

    request = build_life_reflection_action_request(
        tick_id="tick-1",
        reason="idle reflection",
    )

    assert request.kind == "agent_runtime_job"
    assert request.profile_id == "life_reflection.readonly"
    assert request.requires_approval is False
    assert "life_reflection" in request.capabilities_requested
    assert request.denied_capabilities == LIFE_RUNTIME_READONLY_DENIED_CAPABILITIES
    assert "send_message" in request.denied_capabilities
    assert "browser_submit" in request.denied_capabilities
