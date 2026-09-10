"""Runtime service for durable agent jobs."""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Protocol, cast

import structlog

from src.agent_runtime.budget_policy import (
    BudgetDecisionType,
    BudgetJobKind,
    BudgetPolicyDecision,
)
from src.agent_runtime.models import (
    AgentEvent,
    AgentEventType,
    AgentJob,
    AgentJobStatus,
    ContextCapsule,
    ContextPack,
    Finding,
    FindingStatus,
    InvocationProfile,
)

if TYPE_CHECKING:
    from src.agent_runtime.events import AgentEventStream
    from src.agent_runtime.storage import AgentJobStore

logger = structlog.get_logger()


class AgentWorkerBackend(Protocol):
    """Worker backend interface used by Agent Runtime."""

    name: str

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule: ...

    async def cancel(self, job_id: str) -> bool: ...


class AgentMemorySink(Protocol):
    """Receives worker-proposed memory candidates for staging/consolidation."""

    async def stage_candidates(
        self,
        *,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> int: ...


class AgentBudgetPreflightGate(Protocol):
    """Checks budget metadata before a runtime job is created."""

    def evaluate(
        self,
        job_kind: BudgetJobKind,
        *,
        estimated_cost_usd: Decimal,
    ) -> BudgetPolicyDecision: ...


class AgentBudgetUsageRecorder(Protocol):
    """Records budget usage after a runtime job completes."""

    async def record_completed_job(
        self,
        *,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> int: ...


class AgentRuntime:
    """Create, run, recover and cancel Agent Runtime jobs."""

    def __init__(
        self,
        *,
        store: AgentJobStore,
        events: AgentEventStream,
        workers: dict[str, AgentWorkerBackend],
        memory_sink: AgentMemorySink | None = None,
        budget_preflight_gate: AgentBudgetPreflightGate | None = None,
        budget_usage_recorder: AgentBudgetUsageRecorder | None = None,
        max_active_per_chat: int = 1,
        max_queued_per_chat: int = 5,
    ) -> None:
        self.store = store
        self._events = events
        self._workers = dict(workers)
        self._memory_sink = memory_sink
        self._budget_preflight_gate = budget_preflight_gate
        self._budget_usage_recorder = budget_usage_recorder
        self._max_active_per_chat = max_active_per_chat
        self._max_queued_per_chat = max_queued_per_chat
        self._background_tasks: dict[str, asyncio.Task[AgentJob]] = {}

    def register_worker(self, name: str, worker: AgentWorkerBackend) -> None:
        """Register or replace a worker backend after runtime construction."""
        self._workers[name] = worker

    def registered_worker_names(self) -> tuple[str, ...]:
        """Return worker ids physically registered in this runtime process."""
        return tuple(sorted(self._workers))

    async def create_job(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        fingerprint: str,
        kind: str,
        profile: InvocationProfile,
        context_pack: ContextPack,
        status: AgentJobStatus = AgentJobStatus.QUEUED,
    ) -> AgentJob:
        """Create an idempotent active job by source fingerprint."""
        existing = await self.store.find_by_fingerprint(fingerprint)
        if existing is not None and existing.status not in _TERMINAL_STATUSES:
            return existing
        context_pack = self._context_pack_after_budget_preflight(context_pack)
        job = AgentJob.new(
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            fingerprint=fingerprint,
            kind=kind,
            profile=profile,
            context_pack=context_pack,
            status=status,
        )
        await self._enforce_chat_limits(job)
        created = await self.store.create(job)
        await self._emit(
            created.id,
            AgentEventType.CREATED,
            f"Created {kind} job",
            {"profile": profile.id, "worker": profile.worker},
        )
        return created

    async def enqueue(self, job_id: str) -> AgentJob:
        """Mark a draft/waiting job as queued."""
        job = await self.store.get(job_id)
        queued = job.with_status(AgentJobStatus.QUEUED)
        return await self.store.save(queued)

    async def status(self, job_id: str) -> AgentJob:
        """Return current job state."""
        return await self.store.get(job_id)

    def events_for(self, job_id: str) -> list[AgentEvent]:
        """Return curated events for one job."""
        return self._events.events_for(job_id)

    async def emit_progress(
        self,
        job_id: str,
        message: str,
        payload: dict[str, str] | None = None,
    ) -> None:
        """Append a curated progress event for Telegram/audit rendering."""
        await self._emit(job_id, AgentEventType.PROGRESS, message, payload)

    async def attach_followup(self, job_id: str, text: str) -> AgentJob:
        """Attach a user addendum unless the job is terminal."""
        job = await self.store.get(job_id)
        if job.status in _TERMINAL_STATUSES:
            return job
        updated = job.model_copy(update={"followups": (*job.followups, text)})
        saved = await self.store.save(updated)
        await self._emit(job_id, AgentEventType.FOLLOWUP_ATTACHED, text[:200])
        return saved

    async def attach_artifact(self, job_id: str, artifact: str) -> AgentJob:
        """Attach an artifact path unless the job is terminal."""
        job = await self.store.get(job_id)
        if job.status in _TERMINAL_STATUSES:
            return job
        updated = job.model_copy(update={"artifacts": (*job.artifacts, artifact)})
        saved = await self.store.save(updated)
        await self._emit(job_id, AgentEventType.ARTIFACT_ATTACHED, artifact)
        return saved

    async def start(self, job_id: str) -> AgentJob:
        """Run a job with its configured worker backend."""
        job = await self.store.get(job_id)
        if job.status in _TERMINAL_STATUSES:
            return job
        running = await self._mark_running(job)
        return await self._run_worker(running)

    async def start_background(self, job_id: str) -> AgentJob:
        """Start a job in a background asyncio task and return immediately."""
        existing_task = self._background_tasks.get(job_id)
        if existing_task is not None and not existing_task.done():
            return await self.store.get(job_id)
        job = await self.store.get(job_id)
        if job.status in _TERMINAL_STATUSES:
            return job
        running = await self._mark_running(job)
        task = asyncio.create_task(self._run_worker(running))
        self._background_tasks[job_id] = task
        task.add_done_callback(lambda _task: self._background_tasks.pop(job_id, None))
        return running

    async def wait_background(self, job_id: str) -> AgentJob | None:
        """Wait for a background task owned by this process."""
        task = self._background_tasks.get(job_id)
        if task is None:
            job = await self.store.get(job_id)
            return job if job.status in _TERMINAL_STATUSES else None
        try:
            return await task
        except asyncio.CancelledError:
            return await self.store.get(job_id)

    async def cancel(self, job_id: str, *, reason: str = "") -> AgentJob:
        """Cancel a queued/running job and ask worker to stop if possible."""
        job = await self.store.get(job_id)
        worker = self._workers.get(job.profile.worker)
        task = self._background_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        if worker is not None and job.status is AgentJobStatus.RUNNING:
            await worker.cancel(job_id)
        canceled = job.with_status(
            AgentJobStatus.CANCELED,
            error=reason or "canceled",
        )
        await self.store.save(canceled)
        await self._emit(job_id, AgentEventType.CANCELED, canceled.error)
        return canceled

    async def complete(self, job_id: str, result: ContextCapsule) -> AgentJob:
        """Manually complete a job with a capsule."""
        job = await self.store.get(job_id)
        completed = job.with_status(AgentJobStatus.DONE, result=result)
        return await self.store.save(completed)

    async def fail(self, job_id: str, reason: str) -> AgentJob:
        """Mark a job as failed with a reason."""
        job = await self.store.get(job_id)
        failed = job.with_status(AgentJobStatus.FAILED, error=reason)
        return await self.store.save(failed)

    async def recover_running_jobs(self, *, reason: str) -> list[AgentJob]:
        """Move orphaned running jobs into needs_review after bot restart."""
        jobs = await self.store.list_by_status(
            (AgentJobStatus.RUNNING, AgentJobStatus.WAITING_USER)
        )
        recovered: list[AgentJob] = []
        for job in jobs:
            updated = job.with_status(AgentJobStatus.NEEDS_REVIEW, error=reason)
            recovered.append(await self.store.save(updated))
            await self._emit(job.id, AgentEventType.RECOVERED, reason)
        return recovered

    async def _mark_running(self, job: AgentJob) -> AgentJob:
        running_base = job.with_status(AgentJobStatus.RUNNING)
        running = await self.store.save(
            running_base.model_copy(
                update={"observability": self._observability_payload(running_base)}
            )
        )
        await self._emit(
            running.id,
            AgentEventType.STARTED,
            "Agent job started",
            self._observability_payload(running),
        )
        return running

    async def _run_worker(self, running: AgentJob) -> AgentJob:
        worker = self._workers[running.profile.worker]
        try:
            result = await self._run_worker_with_profile_timeout(worker, running)
        except asyncio.CancelledError:
            current = await self.store.get(running.id)
            if current.status in _TERMINAL_STATUSES:
                return current
            canceled = current.with_status(
                AgentJobStatus.CANCELED,
                error="background task canceled",
            )
            await self.store.save(canceled)
            await self._emit(running.id, AgentEventType.CANCELED, canceled.error)
            return canceled
        except TimeoutError:
            reason = _timeout_error_message(running.profile)
            failure_metadata = _timeout_failure_metadata(running)
            failed = running.with_status(
                AgentJobStatus.FAILED,
                error=reason,
                result=_timeout_failure_capsule(running, reason),
            )
            failed = failed.model_copy(
                update={
                    "observability": {
                        **running.observability,
                        **failure_metadata,
                    }
                }
            )
            failed = await self.store.save(failed)
            await self._emit(
                running.id,
                AgentEventType.FAILED,
                reason,
                failure_metadata,
            )
            return failed
        except Exception as exc:
            current = await self.store.get(running.id)
            if current.status in _TERMINAL_STATUSES:
                return current
            failure_metadata = _failure_metadata_from_exception(exc)
            partial_result = _partial_result_from_exception(exc)
            failed = current.with_status(
                AgentJobStatus.FAILED,
                error=str(exc),
                result=partial_result,
            )
            if failure_metadata:
                failed = failed.model_copy(
                    update={
                        "observability": {
                            **current.observability,
                            **failure_metadata,
                        }
                    }
                )
            failed = await self.store.save(failed)
            await self._emit(
                running.id,
                AgentEventType.FAILED,
                str(exc),
                failure_metadata or None,
            )
            return failed

        current = await self.store.get(running.id)
        if current.status in _TERMINAL_STATUSES:
            return current
        completed = running.with_status(AgentJobStatus.DONE, result=result)
        completed = await self.store.save(completed)
        await self._emit(running.id, AgentEventType.COMPLETED, result.summary)
        await self._stage_memory_candidates(completed, result)
        await self._record_budget_usage(completed, result)
        return completed

    async def _run_worker_with_profile_timeout(
        self,
        worker: AgentWorkerBackend,
        running: AgentJob,
    ) -> ContextCapsule:
        timeout_seconds = _profile_timeout_seconds(running.profile)
        if timeout_seconds is None:
            return await self._run_worker_with_optional_progress(worker, running)
        return await asyncio.wait_for(
            self._run_worker_with_optional_progress(worker, running),
            timeout=timeout_seconds,
        )

    async def _run_worker_with_optional_progress(
        self,
        worker: AgentWorkerBackend,
        running: AgentJob,
    ) -> ContextCapsule:
        run_with_progress = cast(
            "Any",
            getattr(worker, "run_with_progress", None),
        )
        if run_with_progress is None:
            return await worker.run(job=running, context_pack=running.context_pack)

        async def progress_callback(message: str) -> None:
            await self.emit_progress(running.id, message)

        return cast(
            "ContextCapsule",
            await run_with_progress(
                job=running,
                context_pack=running.context_pack,
                progress_callback=progress_callback,
            ),
        )

    async def _stage_memory_candidates(
        self,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> None:
        if self._memory_sink is None or not capsule.memory_candidates:
            return
        try:
            count = await self._memory_sink.stage_candidates(
                job=job,
                capsule=capsule,
            )
        except Exception as exc:
            logger.warning(
                "agent_memory_staging_failed",
                job_id=job.id,
                error=str(exc),
                exc_info=True,
            )
            await self._emit(
                job.id,
                AgentEventType.PROGRESS,
                "Memory staging failed",
                {"error": str(exc)[:500]},
            )
            return
        if count > 0:
            await self._emit(
                job.id,
                AgentEventType.MEMORY_STAGED,
                "Memory candidates staged",
                {"count": str(count)},
            )

    async def _record_budget_usage(
        self,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> None:
        if self._budget_usage_recorder is None:
            return
        try:
            count = await self._budget_usage_recorder.record_completed_job(
                job=job,
                capsule=capsule,
            )
        except Exception as exc:
            logger.warning(
                "agent_budget_usage_record_failed",
                job_id=job.id,
                error=str(exc),
                exc_info=True,
            )
            await self._emit(
                job.id,
                AgentEventType.PROGRESS,
                "Budget usage recording failed",
                {"error": str(exc)[:500]},
            )
            return
        if count > 0:
            await self._emit(
                job.id,
                AgentEventType.PROGRESS,
                "Budget usage recorded",
                {"count": str(count)},
            )

    async def _enforce_chat_limits(self, job: AgentJob) -> None:
        active = await self.store.list_by_status(
            (AgentJobStatus.RUNNING, AgentJobStatus.WAITING_USER)
        )
        queued = await self.store.list_by_status(
            (AgentJobStatus.QUEUED, AgentJobStatus.AWAITING_INPUT)
        )
        active_count = sum(1 for item in active if item.chat_id == job.chat_id)
        queued_count = sum(1 for item in queued if item.chat_id == job.chat_id)
        if (
            job.status in {AgentJobStatus.RUNNING, AgentJobStatus.WAITING_USER}
            and active_count >= self._max_active_per_chat
        ):
            raise RuntimeError("active job limit exceeded for chat")
        if (
            job.status in {AgentJobStatus.QUEUED, AgentJobStatus.AWAITING_INPUT}
            and queued_count >= self._max_queued_per_chat
        ):
            raise RuntimeError("queued job limit exceeded for chat")

    def _context_pack_after_budget_preflight(
        self,
        context_pack: ContextPack,
    ) -> ContextPack:
        if self._budget_preflight_gate is None:
            return context_pack
        budget = _budget_metadata(context_pack.metadata)
        if budget is None:
            return context_pack
        job_kind, estimated_cost = budget
        decision = self._budget_preflight_gate.evaluate(
            job_kind,
            estimated_cost_usd=estimated_cost,
        )
        metadata = {
            **context_pack.metadata,
            "budget_decision": decision.decision.value,
            "budget_reason": decision.reason,
        }
        if decision.decision is not BudgetDecisionType.ALLOW:
            raise RuntimeError(f"budget_{decision.reason}")
        return context_pack.model_copy(update={"metadata": metadata})

    async def _emit(
        self,
        job_id: str,
        event_type: AgentEventType,
        message: str,
        payload: dict[str, str] | None = None,
    ) -> None:
        await self._events.emit(
            AgentEvent(
                job_id=job_id,
                event_type=event_type,
                message=message,
                payload=payload or {},
            )
        )

    def _observability_payload(self, job: AgentJob) -> dict[str, str]:
        return {
            "profile": job.profile.id,
            "worker": job.profile.worker,
            "kind": job.kind,
            "allowed_capabilities": ",".join(job.profile.allowed_capabilities),
            "denied_capabilities": ",".join(job.profile.denied_capabilities),
            **job.profile.metadata,
        }


def _timeout_failure_metadata(job: AgentJob) -> dict[str, str]:
    metadata = {
        "stage": "profile_timeout",
        "reason": "profile_timeout",
        "profile": job.profile.id,
        "worker": job.profile.worker,
        "kind": job.kind,
    }
    timeout_seconds = _profile_timeout_seconds(job.profile)
    if timeout_seconds is not None:
        metadata["timeout_seconds"] = f"{timeout_seconds:g}"
    return metadata


def _timeout_failure_capsule(job: AgentJob, reason: str) -> ContextCapsule:
    metadata = _timeout_failure_metadata(job)
    evidence: tuple[str, ...] = (
        f"stage={metadata['stage']}",
        f"reason={metadata['reason']}",
        f"profile={metadata['profile']}",
        f"worker={metadata['worker']}",
        f"kind={metadata['kind']}",
    )
    timeout_seconds = metadata.get("timeout_seconds")
    if timeout_seconds is not None:
        evidence = (*evidence, f"timeout_seconds={timeout_seconds}")
    return ContextCapsule(
        summary=reason,
        findings=(
            Finding(
                claim=(
                    "Agent Runtime stopped the worker because invocation profile "
                    "timeout was reached before a Context Capsule was returned."
                ),
                status=FindingStatus.CONFIRMED,
                confidence=1.0,
                evidence=evidence,
            ),
        ),
        next_actions=(
            "Replay with a larger timeout or split the request into a smaller bounded job.",
        ),
    )


def _failure_metadata_from_exception(exc: Exception) -> dict[str, str]:
    metadata = getattr(exc, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): str(value) for key, value in metadata.items() if value is not None
    }


def _partial_result_from_exception(exc: Exception) -> ContextCapsule | None:
    partial_result = getattr(exc, "partial_result", None)
    if isinstance(partial_result, ContextCapsule):
        return partial_result
    return None


def _profile_timeout_seconds(profile: InvocationProfile) -> float | None:
    raw = profile.metadata.get("timeout_seconds")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _timeout_error_message(profile: InvocationProfile) -> str:
    timeout_seconds = _profile_timeout_seconds(profile)
    if timeout_seconds is None:
        return "agent job timed out"
    return f"agent job timed out after {timeout_seconds:g} seconds"


def _budget_metadata(
    metadata: dict[str, str],
) -> tuple[BudgetJobKind, Decimal] | None:
    raw_kind = metadata.get("budget_job_kind", "").strip()
    if not raw_kind:
        return None
    try:
        job_kind = BudgetJobKind(raw_kind)
    except ValueError as exc:
        raise RuntimeError(f"budget_unknown_job_kind:{raw_kind}") from exc
    raw_cost = metadata.get("budget_estimated_cost_usd", "").strip()
    if not raw_cost:
        raise RuntimeError("budget_estimated_cost_missing")
    try:
        estimated_cost = Decimal(raw_cost)
    except InvalidOperation as exc:
        raise RuntimeError("budget_estimated_cost_invalid") from exc
    if estimated_cost < 0:
        raise RuntimeError("budget_estimated_cost_invalid")
    return job_kind, estimated_cost


_TERMINAL_STATUSES = {
    AgentJobStatus.DONE,
    AgentJobStatus.FAILED,
    AgentJobStatus.CANCELED,
    AgentJobStatus.NEEDS_REVIEW,
}
