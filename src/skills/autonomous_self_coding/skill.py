"""Background skill that schedules Жвуша's autonomous self-coding jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.agent_runtime.models import (
    AgentJobStatus,
    ContextCapsule,
    ContextPack,
    InvocationProfile,
)
from src.agent_runtime.self_work_context import sanitize_self_work_text
from src.skills.base import (
    AgentContext,
    BackgroundSkill,
    ExecutionPlan,
    SideEffect,
    SkillResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.agent_runtime.runtime import AgentRuntime

_ACTIVE_STATUSES = (
    AgentJobStatus.QUEUED,
    AgentJobStatus.AWAITING_INPUT,
    AgentJobStatus.RUNNING,
    AgentJobStatus.WAITING_USER,
)
_SELF_WORK_KINDS = {"self_improvement", "self_coding"}


class SelfWorkContextProvider(Protocol):
    """Optional read-only provider for autonomous self-work planner visibility."""

    async def build_self_work_context_capsule(self) -> ContextCapsule: ...


class AutonomousSelfCodingSkill(BackgroundSkill):
    """Schedule one bounded autonomous self-improvement Agent Runtime job."""

    name: ClassVar[str] = "autonomous_self_coding"
    description: ClassVar[str] = (
        "Autonomous self-work loop that improves Жвуша through gated self-coding"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"
    trigger_type: ClassVar[Literal["cron", "event", "interval"]] = "interval"
    trigger_config: ClassVar[dict[str, object]] = {"seconds": 21600}
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "high"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.DELEGATES_TO_CODE_AGENT,
        SideEffect.CALLS_LLM,
        SideEffect.CALLS_LLM_TIER_STRATEGIST,
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.SPAWNS_SUBPROCESS,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        runtime: AgentRuntime,
        profile: InvocationProfile,
        self_work_context_provider: SelfWorkContextProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._profile = profile
        self._self_work_context_provider = self_work_context_provider
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def set_self_work_context_provider(
        self,
        provider: SelfWorkContextProvider,
    ) -> None:
        """Attach the production context provider after runtime graph startup."""

        self._self_work_context_provider = provider

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del message, context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="background",
            human_summary=(
                "Autonomous self-work: find one safe improvement and run it "
                "through the existing self-coding gates."
            ),
            estimated_tokens=90000,
            estimated_cost_usd=Decimal("1.00"),
            estimated_duration_seconds=3600.0,
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=2,
            delegated_to=self._profile.worker,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        return await self.run_once()

    async def run_once(self, *, cycle_id: str = "") -> SkillResult:
        active = await self._active_self_work_job()
        if active:
            return SkillResult(
                success=True,
                response=(
                    "Autonomous self-coding skipped: active self-work job "
                    f"`{active}` already exists."
                ),
                metadata={"skipped": "active_job", "active_job_id": active},
            )

        source_message_id = cycle_id or self._cycle_id()
        pack = await self._context_pack()
        fingerprint = _fingerprint(
            owner_user_id=self._admin_user_id,
            source_message_id=source_message_id,
            profile_id=self._profile.id,
        )
        job = await self._runtime.create_job(
            owner_user_id=self._admin_user_id,
            chat_id=self._admin_user_id,
            source_message_id=source_message_id,
            fingerprint=fingerprint,
            kind="self_improvement",
            profile=self._profile,
            context_pack=pack,
        )
        running = await self._runtime.start_background(job.id)
        return SkillResult(
            success=True,
            response=f"Autonomous self-coding job started: `{running.id}`.",
            metadata={"agent_job_id": running.id},
        )

    async def wait_background_result(self, job_id: str) -> Any:
        return await self._runtime.wait_background(job_id)

    async def _active_self_work_job(self) -> str:
        for job in await self._runtime.store.list_by_status(_ACTIVE_STATUSES):
            if job.chat_id == self._admin_user_id and job.kind in _SELF_WORK_KINDS:
                return job.id
        return ""

    async def _context_pack(self) -> ContextPack:
        constraints: tuple[str, ...] = (
            "Agent Runtime durable job",
            "self-approve only non-Tier-3 work within configured autonomous max tier",
            "Tier 3 requires Никита chat approval before implementation",
            "Tier 3 approval prompts stay short; details only on request",
            "Tier 3 free-text decisions use AI/classifier, not keyword lists",
            "no live env activation",
            "no restart/publish/browser_submit/send_message",
            "write detailed change summary after completion",
            "preserve /код and ImplementSpec gates",
        )
        active_code_state = ""
        chat_context: tuple[str, ...] = ()
        metadata: dict[str, str] = {}

        if self._self_work_context_provider is not None:
            capsule = (
                await self._self_work_context_provider.build_self_work_context_capsule()
            )
            active_code_state = sanitize_self_work_text(
                capsule.markdown_report or capsule.processed_context
            )
            chat_context = tuple(
                sanitize_self_work_text(item)
                for item in (capsule.summary, *capsule.next_actions)
                if item
            )
            metadata["self_work_context_capsule"] = "true"
            safe_spec_candidates = _safe_spec_candidates(capsule.artifacts)
            if safe_spec_candidates:
                metadata["self_work_safe_spec_candidates"] = "\n".join(
                    safe_spec_candidates
                )
            constraints = (
                *constraints,
                "self-work Context Capsule injected from bounded runtime observations",
            )

        return ContextPack(
            user_request=(
                "Autonomous self-work cycle: discover one low-risk improvement, "
                "create/spec or reuse a safe spec, self-approve only non-Tier-3 "
                "work within the configured mandate, and leave Tier 3 for "
                "Никита's short chat approval."
            ),
            chat_context=chat_context,
            active_code_state=active_code_state,
            constraints=constraints,
            metadata=metadata,
        )

    def _cycle_id(self) -> str:
        now = self._clock().astimezone(UTC)
        bucket = now.strftime("%Y%m%dT%H")
        return f"autonomous-self-coding:{bucket}"


def _fingerprint(
    *,
    owner_user_id: int,
    source_message_id: str,
    profile_id: str,
) -> str:
    digest = sha256(
        f"{owner_user_id}:{source_message_id}:{profile_id}".encode()
    ).hexdigest()
    return f"self-improvement:{digest[:24]}"


def _safe_spec_candidates(artifacts: tuple[str, ...]) -> tuple[str, ...]:
    prefix = "safe_spec_candidate:"
    candidates: list[str] = []
    for artifact in artifacts:
        if not artifact.startswith(prefix):
            continue
        candidate = sanitize_self_work_text(artifact.removeprefix(prefix).strip())
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)
