"""LifeRuntime safety guard contract."""

from __future__ import annotations


def test_life_runtime_safety_blocks_direct_side_effect_capabilities() -> None:
    from src.life_runtime import (
        InnerDecision,
        LifeActionRequest,
        LifeRuntimeSafetyGuard,
    )

    decision = InnerDecision(
        decision_type="propose_agent_job",
        reason="try to send directly",
        action_request=LifeActionRequest(
            requested_by_tick_id="tick-1",
            kind="agent_runtime_job",
            profile_id="bad.send",
            capabilities_requested=("send_message",),
            denied_capabilities=(),
            reason="not allowed",
        ),
    )

    verdict = LifeRuntimeSafetyGuard().evaluate(decision)

    assert verdict.allowed is False
    assert verdict.reason == "side_effect_capability_denied"
    assert verdict.denied_capabilities == ("send_message",)
