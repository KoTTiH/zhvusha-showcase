"""Durable storage tests for Agent Runtime jobs and events."""

from __future__ import annotations


async def test_file_store_persists_jobs_and_supports_restart_recovery(tmp_path) -> None:
    from src.agent_runtime.events import FileAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import FileAgentJobStore

    store = FileAgentJobStore(tmp_path)
    runtime = AgentRuntime(
        store=store,
        events=FileAgentEventStream(tmp_path),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:100",
        fingerprint="persisted",
        kind="source_compare",
        profile=InvocationProfile(id="readonly"),
        context_pack=ContextPack(user_request="recover"),
    )
    await store.save(job.with_status(AgentJobStatus.RUNNING))

    recovered_runtime = AgentRuntime(
        store=FileAgentJobStore(tmp_path),
        events=FileAgentEventStream(tmp_path),
        workers={},
    )
    recovered = await recovered_runtime.recover_running_jobs(reason="restart")

    assert recovered[0].id == job.id
    assert recovered[0].status is AgentJobStatus.NEEDS_REVIEW


async def test_file_event_stream_persists_curated_events(tmp_path) -> None:
    from src.agent_runtime.events import FileAgentEventStream
    from src.agent_runtime.models import AgentEvent, AgentEventType

    stream = FileAgentEventStream(tmp_path)
    await stream.emit(
        AgentEvent(
            job_id="job-1",
            event_type=AgentEventType.PROGRESS,
            message="Собираю контекст проекта.",
            payload={"profile": "source_compare"},
        )
    )

    events = FileAgentEventStream(tmp_path).events_for("job-1")

    assert events[0].message == "Собираю контекст проекта."
    assert events[0].payload["profile"] == "source_compare"
