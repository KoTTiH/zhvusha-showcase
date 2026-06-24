"""Busy/follow-up routing tests for active Agent Runtime jobs."""

from __future__ import annotations


async def test_active_job_status_question_returns_curated_status() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentEventType,
        AgentJobStatus,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.routing import (
        AgentMessageIntent,
        route_message_for_active_job,
    )
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))
    await runtime.emit_progress(job.id, "Сверяю пост с кодом.")

    decision = await route_message_for_active_job(
        runtime,
        chat_id=2,
        text="что там, зависла?",
    )

    assert decision.intent is AgentMessageIntent.STATUS_QUERY
    assert decision.job_id == job.id
    assert "Сверяю пост с кодом" in decision.reply
    assert AgentEventType.PROGRESS.value in {
        event.event_type.value for event in runtime.events_for(job.id)
    }


async def test_awaiting_input_job_treats_next_message_as_followup() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.routing import (
        AgentMessageIntent,
        route_message_for_active_job,
    )
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сейчас скину пост"),
        status=AgentJobStatus.AWAITING_INPUT,
    )

    decision = await route_message_for_active_job(
        runtime,
        chat_id=2,
        text="Anthropic представил режим сновидений для агентов",
    )

    updated = await runtime.status(job.id)

    assert decision.intent is AgentMessageIntent.FOLLOWUP
    assert updated.followups == ("Anthropic представил режим сновидений для агентов",)


async def test_active_job_separates_new_topic_from_followup() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.routing import (
        AgentMessageIntent,
        route_message_for_active_job,
    )
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))

    decision = await route_message_for_active_job(
        runtime,
        chat_id=2,
        text="новая тема: придумай пост для канала",
    )

    updated = await runtime.status(job.id)

    assert decision.intent is AgentMessageIntent.NEW_TASK
    assert decision.reply
    assert updated.followups == ()


async def test_active_job_cancel_request_cancels_job() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.routing import (
        AgentMessageIntent,
        route_message_for_active_job,
    )
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))

    decision = await route_message_for_active_job(
        runtime,
        chat_id=2,
        text="отмени agent задачу",
    )

    updated = await runtime.status(job.id)

    assert decision.intent is AgentMessageIntent.CANCEL
    assert updated.status is AgentJobStatus.CANCELED
    assert "canceled" in decision.reply
