"""Bridge helpers from AgencyIntent to Agent Runtime context packs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agent_runtime.models import ContextPack

if TYPE_CHECKING:
    from src.agency.models import AgencyIntent, AutonomyPolicyDecision


def build_agency_context_pack(
    intent: AgencyIntent,
    *,
    policy_decision: AutonomyPolicyDecision | None = None,
) -> ContextPack:
    """Serialize an agency intent into a bounded Agent Runtime context pack."""

    metadata = {
        "agency_intent_id": intent.id,
        "agency_intent_json": intent.model_dump_json(),
    }
    if policy_decision is not None:
        metadata["agency_policy_decision_json"] = policy_decision.model_dump_json()

    return ContextPack(
        user_request=intent.goal,
        constraints=(
            "Agency job is read-only/draft/staging by default.",
            "External side effects require policy and approval gates.",
            *intent.safety_constraints,
        ),
        metadata=metadata,
    )
