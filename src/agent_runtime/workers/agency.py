"""Agent Runtime worker for bounded agency intents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agency.models import AgencyIntent, AutonomyPolicyDecision
from src.agency.rendering import (
    render_agency_intent_status,
    render_policy_decision_status,
)
from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextPack


class AgencyWorkerBackend:
    """Read-only/draft worker for one AgencyIntent.

    This MVP worker does not execute tools. It turns a prebuilt intent into a
    Context Capsule so Жвуша can inspect next actions and memory candidates.
    """

    name = "agency"

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        del job
        intent = AgencyIntent.model_validate_json(
            context_pack.metadata["agency_intent_json"]
        )
        policy = _policy_from_context(context_pack)
        next_actions = _next_actions(intent, policy)
        summary = f"AgencyIntent `{intent.id}` prepared: {intent.goal}"
        report = _render_report(intent, policy)
        return ContextCapsule(
            summary=summary,
            processed_context=intent.model_dump_json(),
            findings=(
                Finding(
                    claim="AgencyIntent was processed without executing tools.",
                    status=FindingStatus.CONFIRMED,
                    confidence=1.0,
                    evidence=(intent.id,),
                ),
            ),
            sources=intent.evidence,
            memory_candidates=(f"Agency intent `{intent.kind.value}`: {intent.goal}",),
            next_actions=next_actions or ("No candidate actions.",),
            markdown_report=report,
        )

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False


def _policy_from_context(context_pack: ContextPack) -> AutonomyPolicyDecision | None:
    raw = context_pack.metadata.get("agency_policy_decision_json")
    if not raw:
        return None
    return AutonomyPolicyDecision.model_validate_json(raw)


def _next_actions(
    intent: AgencyIntent,
    policy: AutonomyPolicyDecision | None,
) -> tuple[str, ...]:
    if policy is None:
        return tuple(
            f"{action.kind.value}: {action.description or action.capability}"
            for action in intent.candidate_actions
        )
    items: list[str] = [
        f"policy:{policy.decision.value}: {policy.reason}",
    ]
    items.extend(
        f"allowed:{action.kind.value}: {action.description or action.capability}"
        for action in policy.allowed_actions
    )
    items.extend(
        f"blocked:{action.kind.value}: {action.description or action.capability}"
        for action in policy.blocked_actions
    )
    if policy.permission_request is not None:
        items.append(
            "ask_nikita:"
            f"{policy.permission_request.target_id}:"
            f"{','.join(scope.value for scope in policy.permission_request.requested_scopes)}"
        )
    return tuple(items) or ("No candidate actions.",)


def _render_report(
    intent: AgencyIntent,
    policy: AutonomyPolicyDecision | None,
) -> str:
    parts = [render_agency_intent_status(intent)]
    if policy is not None:
        parts.extend(("", render_policy_decision_status(policy)))
    return "\n".join(parts)
