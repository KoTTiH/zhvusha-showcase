"""Safety guards for bounded LifeRuntime decisions."""

from __future__ import annotations

from src.agent_runtime.tools import SIDE_EFFECT_CAPABILITIES
from src.life_runtime.models import InnerDecision, LifeRuntimeSafetyVerdict

_LIFE_RUNTIME_FORBIDDEN_CAPABILITIES = frozenset(
    (
        *SIDE_EFFECT_CAPABILITIES,
        "write_whitelisted_files_after_approval",
        "commit_after_gate",
        "browser_draft_form",
    )
)


class LifeRuntimeSafetyGuard:
    """Fail closed before LifeRuntime can request side effects."""

    def evaluate(self, decision: InnerDecision) -> LifeRuntimeSafetyVerdict:
        """Evaluate one inner decision against MVP read-only constraints."""

        request = decision.action_request
        if request is None:
            return LifeRuntimeSafetyVerdict(allowed=True, reason="no_action_request")
        requested = set(request.capabilities_requested)
        forbidden = tuple(
            sorted(requested.intersection(_LIFE_RUNTIME_FORBIDDEN_CAPABILITIES))
        )
        if forbidden:
            return LifeRuntimeSafetyVerdict(
                allowed=False,
                reason="side_effect_capability_denied",
                denied_capabilities=forbidden,
            )
        if request.requires_approval:
            return LifeRuntimeSafetyVerdict(
                allowed=False,
                reason="approval_required_not_executed_by_life_runtime",
                denied_capabilities=request.capabilities_requested,
            )
        return LifeRuntimeSafetyVerdict(
            allowed=True,
            reason="readonly_life_runtime_request",
        )
