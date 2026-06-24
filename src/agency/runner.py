"""Agency runner that bridges Жвушин intent into Agent Runtime jobs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.agency.agent_runtime_bridge import build_agency_context_pack
from src.agent_runtime.profiles import AGENCY_READONLY_DRAFT

if TYPE_CHECKING:
    from datetime import datetime

    from src.agency.intent_builder import PersonalityDrivenIntentBuilder
    from src.agency.models import (
        AgencyIntent,
        AgencyPermissionRequest,
        AutonomyPolicyDecision,
        SocialPermissionGrant,
    )
    from src.agency.policy import AutonomyPolicy
    from src.agent_runtime.models import AgentJob, ContextCapsule
    from src.agent_runtime.runtime import AgentRuntime
    from src.daemon.signals import Signal
    from src.memory.protocols import LearningSignal
    from src.personality.protocols import AffectiveSnapshot, HomeostasisCorrection


@dataclass(frozen=True)
class AgencyRunResult:
    """Structured result returned to Жвушин orchestrator."""

    intent: AgencyIntent
    policy_decision: AutonomyPolicyDecision
    job: AgentJob
    capsule: ContextCapsule | None = None
    next_actions: tuple[str, ...] = ()
    permission_request: AgencyPermissionRequest | None = None


class AgencyRunner:
    """Build intent, evaluate policy and run one bounded agency job."""

    def __init__(
        self,
        *,
        builder: PersonalityDrivenIntentBuilder,
        policy: AutonomyPolicy,
        runtime: AgentRuntime,
    ) -> None:
        self._builder = builder
        self._policy = policy
        self._runtime = runtime

    async def run_once(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        event: str,
        affective_snapshot: AffectiveSnapshot | None = None,
        homeostasis_corrections: tuple[HomeostasisCorrection, ...] = (),
        desire_signals: tuple[str, ...] = (),
        learning_signals: tuple[LearningSignal, ...] = (),
        daemon_signals: tuple[Signal, ...] = (),
        memory_evidence: tuple[str, ...] = (),
        grants: tuple[SocialPermissionGrant, ...] = (),
        now: datetime | None = None,
    ) -> AgencyRunResult:
        """Run a safe read-only/draft agency pass and return its capsule."""

        intent = self._builder.build(
            event=event,
            affective_snapshot=affective_snapshot,
            homeostasis_corrections=homeostasis_corrections,
            desire_signals=desire_signals,
            learning_signals=learning_signals,
            daemon_signals=daemon_signals,
            memory_evidence=memory_evidence,
        )
        decision = self._policy.decide(intent, grants=grants, now=now)
        context_pack = build_agency_context_pack(
            intent,
            policy_decision=decision,
        )
        job = await self._runtime.create_job(
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            fingerprint=_fingerprint(
                chat_id=chat_id,
                source_message_id=source_message_id,
                intent=intent,
            ),
            kind="agency",
            profile=AGENCY_READONLY_DRAFT,
            context_pack=context_pack,
        )
        completed = await self._runtime.start(job.id)
        capsule = completed.result
        return AgencyRunResult(
            intent=intent,
            policy_decision=decision,
            job=completed,
            capsule=capsule,
            next_actions=capsule.next_actions if capsule is not None else (),
            permission_request=decision.permission_request,
        )


def _fingerprint(
    *,
    chat_id: int,
    source_message_id: str,
    intent: AgencyIntent,
) -> str:
    digest = hashlib.sha256(
        "|".join(
            (
                intent.source,
                intent.goal,
                ",".join(intent.personality_drivers),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"agency:{chat_id}:{source_message_id}:{digest}"
