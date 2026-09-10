"""Agent Runtime worker for autonomous Жвуша self-improvement cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.skills.autonomous_self_coding.planner import (  # noqa: TC001
    SelfImprovementCycleResult,
)

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextPack


class SelfImprovementEngine(Protocol):
    """Narrow engine protocol used by the worker."""

    async def run_once(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> SelfImprovementCycleResult: ...


class AutonomousSelfCodingWorkerBackend:
    """Bounded worker that delegates planning to the self-coding engine."""

    name = "self_improvement"

    def __init__(self, *, engine: SelfImprovementEngine) -> None:
        self._engine = engine

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        result = await self._engine.run_once(job=job, context_pack=context_pack)
        artifacts: list[str] = []
        if result.spec_slug:
            artifacts.append(result.spec_slug)
            artifacts.append(f"spec_slug: {result.spec_slug}")
        if result.implementation_job_id:
            artifacts.append(result.implementation_job_id)
            artifacts.append(f"implementation_job_id: {result.implementation_job_id}")
        if result.change_summary_path:
            artifacts.append(f"change_summary_path: {result.change_summary_path}")
        if result.needs_user_confirmation:
            artifacts.append("needs_user_confirmation:true")
        return ContextCapsule(
            summary=result.summary,
            processed_context=result.details,
            findings=(
                Finding(
                    claim=f"Autonomous self-improvement status: {result.status}",
                    status=FindingStatus.CONFIRMED,
                    confidence=1.0,
                    evidence=tuple(artifacts),
                ),
            ),
            sources=(f"tasks/{result.spec_slug}.yaml",) if result.spec_slug else (),
            artifacts=tuple(artifacts),
            memory_candidates=result.memory_candidates,
            next_actions=result.next_actions,
            markdown_report=result.summary,
        )

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False
