"""Agent Runtime service tests."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest


async def test_runtime_creates_job_and_deduplicates_by_fingerprint() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    profile = InvocationProfile(id="readonly", allowed_capabilities=("read_code",))

    first = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:10",
        fingerprint="same",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи пост"),
    )
    second = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:10",
        fingerprint="same",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи пост"),
    )

    assert second.id == first.id


async def test_runtime_reuses_fingerprint_only_while_previous_job_is_non_terminal() -> (
    None
):
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextCapsule, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    profile = InvocationProfile(id="readonly", allowed_capabilities=("read_code",))

    first = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:10",
        fingerprint="same",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи пост"),
    )
    duplicate = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:10",
        fingerprint="same",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи пост"),
    )
    await runtime.complete(first.id, ContextCapsule(summary="готово"))

    retry = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:11",
        fingerprint="same",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи пост снова"),
    )

    assert duplicate.id == first.id
    assert retry.id != first.id
    assert retry.fingerprint == "same"


async def test_runtime_start_is_noop_for_terminal_jobs() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeWorker(AgentWorkerBackend):
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            self.calls += 1
            return ContextCapsule(summary="готово")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    worker = FakeWorker()
    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"fake": worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:11",
        fingerprint="terminal-start",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="fake"),
        context_pack=ContextPack(user_request="сравни"),
    )
    completed = await runtime.start(job.id)

    restarted = await runtime.start(job.id)

    assert completed.status is AgentJobStatus.DONE
    assert restarted.id == completed.id
    assert restarted.status is AgentJobStatus.DONE
    assert worker.calls == 1
    assert [event.event_type.value for event in events.events_for(job.id)] == [
        "created",
        "started",
        "completed",
    ]


async def test_runtime_applies_profile_timeout_to_worker() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentEventType,
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class SlowWorker(AgentWorkerBackend):
        name = "slow"

        def __init__(self) -> None:
            self.cancelled = False

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return ContextCapsule(summary="too late")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    worker = SlowWorker()
    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"slow": worker},
    )
    profile = InvocationProfile(
        id="slow.timeout",
        worker="slow",
        metadata={"timeout_seconds": "0.01"},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:timeout",
        fingerprint="timeout",
        kind="source_compare",
        profile=profile,
        context_pack=ContextPack(user_request="изучи долго"),
    )

    completed = await runtime.start(job.id)
    failed_events = [
        event
        for event in events.events_for(job.id)
        if event.event_type is AgentEventType.FAILED
    ]

    assert completed.status is AgentJobStatus.FAILED
    assert completed.error == "agent job timed out after 0.01 seconds"
    assert worker.cancelled is True
    assert completed.result is not None
    assert completed.result.summary == "agent job timed out after 0.01 seconds"
    assert completed.result.findings[0].status.value == "confirmed"
    assert "invocation profile timeout" in completed.result.findings[0].claim
    assert completed.observability["stage"] == "profile_timeout"
    assert completed.observability["reason"] == "profile_timeout"
    assert completed.observability["timeout_seconds"] == "0.01"
    assert len(failed_events) == 1
    assert failed_events[0].payload == {
        "stage": "profile_timeout",
        "reason": "profile_timeout",
        "profile": "slow.timeout",
        "worker": "slow",
        "kind": "source_compare",
        "timeout_seconds": "0.01",
    }


async def test_runtime_preserves_source_compare_partial_result_on_timeout() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentEventType,
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.workers.source_compare import SourceCompareWorkerBackend

    class SlowCodeWorker(AgentWorkerBackend):
        name = "codex_cli"

        def __init__(self) -> None:
            self.cancelled = False

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return ContextCapsule(summary="too late")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    class SourceWorker(AgentWorkerBackend):
        name = "web_research"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            return ContextCapsule(
                summary="source context preserved",
                processed_context="source text",
                sources=("https://example.com/post",),
            )

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    code_worker = SlowCodeWorker()
    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={
            "source_compare": SourceCompareWorkerBackend(
                code_worker=code_worker,
                web_worker=SourceWorker(),
            ),
        },
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:source-timeout",
        fingerprint="source-timeout",
        kind="source_compare",
        profile=InvocationProfile(
            id="source_compare.timeout",
            worker="source_compare",
            metadata={"timeout_seconds": "0.01"},
        ),
        context_pack=ContextPack(
            user_request="Сравни проект с https://example.com/post"
        ),
    )

    completed = await runtime.start(job.id)
    failed_events = [
        event
        for event in events.events_for(job.id)
        if event.event_type is AgentEventType.FAILED
    ]

    assert completed.status is AgentJobStatus.FAILED
    assert (
        completed.error
        == "source_compare code analysis canceled after source context read"
    )
    assert code_worker.cancelled is True
    assert completed.result is not None
    assert completed.result.summary == "source context preserved"
    assert completed.result.processed_context == "source text"
    assert completed.observability["stage"] == "code_analysis"
    assert completed.observability["reason"] == "runtime_timeout_or_cancellation"
    assert completed.observability["partial_result"] == "source context preserved"
    assert len(failed_events) == 1
    assert failed_events[0].payload == {
        "stage": "code_analysis",
        "reason": "runtime_timeout_or_cancellation",
        "partial_result": "source context preserved",
    }


async def test_runtime_runs_worker_and_stores_context_capsule() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeWorker(AgentWorkerBackend):
        name = "fake"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            assert job.profile.allows("read_code")
            return ContextCapsule(
                summary="готово",
                processed_context=context_pack.user_request,
                markdown_report="## готово",
            )

        async def cancel(self, job_id: str) -> bool:
            return True

    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"fake": FakeWorker()},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:11",
        fingerprint="run",
        kind="source_compare",
        profile=InvocationProfile(
            id="readonly",
            worker="fake",
            allowed_capabilities=("read_code",),
        ),
        context_pack=ContextPack(user_request="сравни"),
    )

    completed = await runtime.start(job.id)

    assert completed.status is AgentJobStatus.DONE
    assert completed.result is not None
    assert completed.result.summary == "готово"
    assert completed.observability["profile"] == "readonly"
    assert completed.observability["worker"] == "fake"
    assert [event.event_type.value for event in events.events_for(job.id)] == [
        "created",
        "started",
        "completed",
    ]


async def test_runtime_budget_preflight_blocks_budgeted_job_creation() -> None:
    from src.agent_runtime.budget_policy import (
        BudgetDecisionType,
        BudgetJobKind,
        BudgetPolicyDecision,
        BudgetRoute,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class BlockedGate:
        def evaluate(
            self,
            job_kind: BudgetJobKind,
            *,
            estimated_cost_usd: Decimal,
        ) -> BudgetPolicyDecision:
            assert job_kind is BudgetJobKind.READONLY_RESEARCH
            assert estimated_cost_usd == Decimal("0.02")
            return BudgetPolicyDecision(
                decision=BudgetDecisionType.BLOCK,
                reason="daily_budget_exceeded",
                route=BudgetRoute(
                    job_kind=BudgetJobKind.READONLY_RESEARCH,
                    tier="worker",
                    provider="codex_cli",
                    model="default",
                    reasoning_effort="medium",
                    daily_budget_usd=Decimal("0.10"),
                    weekly_budget_usd=Decimal("1.00"),
                    auto_run_allowed=True,
                ),
                status_lines=("blocked by runtime budget",),
            )

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
        budget_preflight_gate=BlockedGate(),
    )

    with pytest.raises(RuntimeError, match="budget_daily_budget_exceeded"):
        await runtime.create_job(
            owner_user_id=1,
            chat_id=2,
            source_message_id="tg:budget",
            fingerprint="budget-blocked",
            kind="web_research",
            profile=InvocationProfile(id="web_research.readonly"),
            context_pack=ContextPack(
                user_request="research",
                metadata={
                    "budget_job_kind": "readonly_research",
                    "budget_estimated_cost_usd": "0.02",
                },
            ),
        )


async def test_runtime_records_budget_usage_after_budgeted_job_completion(
    tmp_path,
) -> None:
    from src.agent_runtime.budget_policy import (
        AgentRuntimeBudgetUsageRecorder,
        BudgetJobKind,
        FileBudgetUsageLedger,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentEventType,
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeWorker(AgentWorkerBackend):
        name = "fake"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            return ContextCapsule(summary="готово")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    ledger = FileBudgetUsageLedger(tmp_path / "budget-usage.jsonl")
    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"fake": FakeWorker()},
        budget_usage_recorder=AgentRuntimeBudgetUsageRecorder(ledger=ledger),
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:budget-record",
        fingerprint="budget-record",
        kind="web_research",
        profile=InvocationProfile(id="web_research.readonly", worker="fake"),
        context_pack=ContextPack(
            user_request="research",
            metadata={
                "budget_job_kind": "readonly_research",
                "budget_estimated_cost_usd": "0.03",
            },
        ),
    )

    await runtime.start(job.id)

    records = ledger.list_records()
    assert len(records) == 1
    assert records[0].job_kind is BudgetJobKind.READONLY_RESEARCH
    assert records[0].cost_usd == Decimal("0.03")
    assert records[0].job_id == job.id
    assert any(
        event.event_type is AgentEventType.PROGRESS
        and event.message == "Budget usage recorded"
        for event in events.events_for(job.id)
    )


async def test_runtime_records_worker_progress_events() -> None:
    from collections.abc import Awaitable, Callable

    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentEventType,
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class ProgressWorker(AgentWorkerBackend):
        name = "progress"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            raise AssertionError("runtime should prefer run_with_progress")

        async def run_with_progress(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
            progress_callback: Callable[[str], Awaitable[None]],
        ) -> ContextCapsule:
            del job, context_pack
            await progress_callback("Сверяю источник с кодом.")
            return ContextCapsule(summary="готово")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"progress": ProgressWorker()},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:progress",
        fingerprint="progress",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="progress"),
        context_pack=ContextPack(user_request="сравни"),
    )

    await runtime.start(job.id)

    assert [
        event.message
        for event in events.events_for(job.id)
        if event.event_type is AgentEventType.PROGRESS
    ] == ["Сверяю источник с кодом."]


async def test_runtime_stages_memory_candidates_through_sink() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class FakeWorker(AgentWorkerBackend):
        name = "fake"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            return ContextCapsule(
                summary="готово",
                memory_candidates=("source_compare нашёл полезный паттерн",),
            )

        async def cancel(self, job_id: str) -> bool:
            return False

    class Sink:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        async def stage_candidates(
            self,
            *,
            job: AgentJob,
            capsule: ContextCapsule,
        ) -> int:
            self.calls.append((job.id, capsule.memory_candidates))
            return len(capsule.memory_candidates)

    sink = Sink()
    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={"fake": FakeWorker()},
        memory_sink=sink,
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:memory",
        fingerprint="memory",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="fake"),
        context_pack=ContextPack(user_request="сравни"),
    )

    await runtime.start(job.id)

    assert sink.calls == [(job.id, ("source_compare нашёл полезный паттерн",))]
    assert [event.event_type.value for event in events.events_for(job.id)] == [
        "created",
        "started",
        "completed",
        "memory_staged",
    ]


async def test_runtime_can_start_job_in_background() -> None:
    import asyncio

    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class SlowWorker(AgentWorkerBackend):
        name = "slow"

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            await self.release.wait()
            return ContextCapsule(summary="done", processed_context="")

        async def cancel(self, job_id: str) -> bool:
            return True

    worker = SlowWorker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"slow": worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:bg",
        fingerprint="bg",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="slow"),
        context_pack=ContextPack(user_request="background"),
    )

    running = await runtime.start_background(job.id)
    worker.release.set()
    completed = await runtime.wait_background(job.id)

    assert running.status is AgentJobStatus.RUNNING
    assert completed is not None
    assert completed.status is AgentJobStatus.DONE


async def test_recover_marks_running_jobs_as_needs_review() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    store = InMemoryAgentJobStore()
    runtime = AgentRuntime(store=store, events=InMemoryAgentEventStream(), workers={})
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:12",
        fingerprint="recover",
        kind="source_compare",
        profile=InvocationProfile(id="readonly"),
        context_pack=ContextPack(user_request="recover me"),
    )
    await store.save(job.with_status(AgentJobStatus.RUNNING))

    recovered = await runtime.recover_running_jobs(reason="bot restarted")

    assert recovered[0].status is AgentJobStatus.NEEDS_REVIEW
    assert recovered[0].error == "bot restarted"


async def test_recover_preserves_queued_and_marks_waiting_user_for_review() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    store = InMemoryAgentJobStore()
    runtime = AgentRuntime(store=store, events=InMemoryAgentEventStream(), workers={})
    queued = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:30",
        fingerprint="queued",
        kind="source_compare",
        profile=InvocationProfile(id="readonly"),
        context_pack=ContextPack(user_request="queued"),
    )
    waiting = await runtime.create_job(
        owner_user_id=1,
        chat_id=3,
        source_message_id="tg:31",
        fingerprint="waiting",
        kind="source_compare",
        profile=InvocationProfile(id="readonly"),
        context_pack=ContextPack(user_request="waiting"),
    )
    await store.save(waiting.with_status(AgentJobStatus.WAITING_USER))

    recovered = await runtime.recover_running_jobs(reason="bot restarted")

    assert await runtime.status(queued.id) == queued
    assert recovered[0].id == waiting.id
    assert recovered[0].status is AgentJobStatus.NEEDS_REVIEW


async def test_cancel_running_job_calls_worker_and_blocks_followups() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class CancelWorker(AgentWorkerBackend):
        name = "cancel"

        def __init__(self) -> None:
            self.cancelled: list[str] = []

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            return ContextCapsule(summary="never", processed_context="")

        async def cancel(self, job_id: str) -> bool:
            self.cancelled.append(job_id)
            return True

    worker = CancelWorker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"cancel": worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:13",
        fingerprint="cancel",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="cancel"),
        context_pack=ContextPack(user_request="cancel me"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))

    canceled = await runtime.cancel(job.id, reason="Никита отменил")
    after_followup = await runtime.attach_followup(job.id, "добавь это")

    assert worker.cancelled == [job.id]
    assert canceled.status is AgentJobStatus.CANCELED
    assert after_followup.status is AgentJobStatus.CANCELED
    assert after_followup.followups == ()


async def test_cancel_background_job_does_not_complete_after_cancel() -> None:
    import asyncio

    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class WaitingWorker(AgentWorkerBackend):
        name = "waiting"

        def __init__(self) -> None:
            self.release = asyncio.Event()
            self.cancelled: list[str] = []

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            await self.release.wait()
            return ContextCapsule(summary="should not win", processed_context="")

        async def cancel(self, job_id: str) -> bool:
            self.cancelled.append(job_id)
            return True

    worker = WaitingWorker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"waiting": worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:bg-cancel",
        fingerprint="bg-cancel",
        kind="source_compare",
        profile=InvocationProfile(id="readonly", worker="waiting"),
        context_pack=ContextPack(user_request="background"),
    )
    await runtime.start_background(job.id)

    canceled = await runtime.cancel(job.id, reason="stop")
    waited = await runtime.wait_background(job.id)

    assert worker.cancelled == [job.id]
    assert canceled.status is AgentJobStatus.CANCELED
    assert waited is not None
    assert waited.status is AgentJobStatus.CANCELED


async def test_runtime_enforces_per_chat_queue_limit() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
        max_queued_per_chat=1,
    )

    await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:20",
        fingerprint="first",
        kind="source_compare",
        profile=InvocationProfile(id="readonly"),
        context_pack=ContextPack(user_request="first"),
    )

    import pytest

    with pytest.raises(RuntimeError, match="queued job limit"):
        await runtime.create_job(
            owner_user_id=1,
            chat_id=2,
            source_message_id="tg:21",
            fingerprint="second",
            kind="source_compare",
            profile=InvocationProfile(id="readonly"),
            context_pack=ContextPack(user_request="second"),
        )
