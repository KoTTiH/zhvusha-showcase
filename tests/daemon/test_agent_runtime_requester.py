"""Daemon -> Agent Runtime job requester contract."""

from __future__ import annotations


def test_daemon_requester_disabled_by_default_and_allows_readonly_when_enabled() -> (
    None
):
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
    )

    request = DaemonAgentRuntimeJobRequest(
        kind="web_research",
        profile_id="web_research.readonly",
        user_request="read-only research",
        source_signal_id="signal-1",
    )

    disabled = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES
    ).plan(request)
    enabled = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
    ).plan(request)

    assert disabled.allowed is False
    assert disabled.reason == "requester_disabled"
    assert enabled.allowed is True
    assert enabled.context_pack is not None
    assert enabled.profile is not None
    assert enabled.profile.id == "web_research.readonly"
    assert (
        enabled.context_pack.metadata["daemon_source_signal_id"]
        == request.source_signal_id
    )
    assert "daemon_requester_readonly_preflight" in enabled.context_pack.constraints


def test_daemon_requester_blocks_side_effect_profiles() -> None:
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
    )

    requester = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
    )

    blocked = requester.plan(
        DaemonAgentRuntimeJobRequest(
            kind="telegram_mcp",
            profile_id="telegram_mcp.personal_actions",
            user_request="send something",
            source_signal_id="signal-2",
        )
    )

    assert blocked.allowed is False
    assert blocked.reason == "profile_not_allowed_for_daemon"


def test_daemon_requester_blocks_side_effect_capabilities_even_if_allowlisted() -> None:
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
    )

    requester = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
        allowed_profile_ids=("telegram_mcp.personal_actions",),
    )

    blocked = requester.plan(
        DaemonAgentRuntimeJobRequest(
            kind="telegram_mcp",
            profile_id="telegram_mcp.personal_actions",
            user_request="send something",
            source_signal_id="signal-3",
        )
    )

    assert blocked.allowed is False
    assert blocked.can_enqueue is False
    assert blocked.reason.startswith("side_effect_capability_denied:")
    assert "telegram_mcp_send" in blocked.reason


def test_daemon_requester_contract_does_not_embed_budget_prices() -> None:
    from src.daemon.agent_runtime_requester import DaemonAgentRuntimeJobRequest

    fields = set(DaemonAgentRuntimeJobRequest.model_fields)

    assert "estimated_cost_usd" not in fields
    assert "budget_job_kind" not in fields


def test_daemon_requester_plan_status_report_shows_readiness_without_enqueue() -> None:
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
        render_daemon_agent_runtime_plan_status,
    )

    requester = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
    )
    plan = requester.plan(
        DaemonAgentRuntimeJobRequest(
            kind="web_research",
            profile_id="web_research.readonly",
            user_request="Проверить источники по runtime.",
            source_signal_id="signal-runtime",
            fingerprint="daemon-fp",
        )
    )

    status = render_daemon_agent_runtime_plan_status(plan)

    assert "Daemon Agent Runtime request: ready" in status
    assert "allowed: yes" in status
    assert "can_enqueue: yes" in status
    assert "kind: web_research" in status
    assert "profile: web_research.readonly" in status
    assert "source_signal_id: signal-runtime" in status
    assert "constraints: daemon_requester_readonly_preflight" in status
    assert "Проверить источники по runtime." not in status


def test_daemon_requester_plan_status_report_shows_block_reason() -> None:
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
        render_daemon_agent_runtime_plan_status,
    )

    plan = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES
    ).plan(
        DaemonAgentRuntimeJobRequest(
            kind="web_research",
            profile_id="web_research.readonly",
            user_request="Проверить источники по runtime.",
            source_signal_id="signal-runtime",
        )
    )

    status = render_daemon_agent_runtime_plan_status(plan)

    assert "Daemon Agent Runtime request: requester_disabled" in status
    assert "allowed: no" in status
    assert "can_enqueue: no" in status
    assert "profile: missing" in status
    assert "Проверить источники по runtime." not in status


async def test_daemon_requester_enqueue_creates_queued_runtime_job_without_starting() -> (
    None
):
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentEventType, AgentJobStatus
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequest,
        DaemonAgentRuntimeJobRequester,
        render_daemon_agent_runtime_enqueue_status,
    )

    events = InMemoryAgentEventStream()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=events,
        workers={},
    )
    requester = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
    )
    request = DaemonAgentRuntimeJobRequest(
        kind="web_research",
        profile_id="web_research.readonly",
        user_request="Проверить источники по runtime.",
        source_signal_id="signal-runtime",
        source_message_id="daemon:signal-runtime",
        chat_id=100,
        owner_user_id=200,
    )

    result = await requester.enqueue(request, runtime)
    duplicate = await requester.enqueue(request, runtime)

    assert result.allowed is True
    assert result.job_id
    assert duplicate.job_id == result.job_id
    job = await runtime.status(result.job_id)
    assert job.status is AgentJobStatus.QUEUED
    assert job.kind == "web_research"
    assert job.profile.id == "web_research.readonly"
    assert job.chat_id == 100
    assert job.owner_user_id == 200
    assert "daemon_requester_readonly_preflight" in job.context_pack.constraints
    assert job.context_pack.metadata["daemon_source_signal_id"] == "signal-runtime"
    assert [event.event_type for event in events.events_for(job.id)] == [
        AgentEventType.CREATED
    ]

    status = render_daemon_agent_runtime_enqueue_status(result)

    assert "Daemon Agent Runtime enqueue: queued" in status
    assert f"job_id: {result.job_id}" in status
    assert "Проверить источники по runtime." not in status


async def test_daemon_agent_runtime_tool_enqueues_allowed_readonly_job() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.daemon.agent_runtime_requester import DaemonAgentRuntimeJobRequester
    from src.daemon.tools.agent_runtime import AgentRuntimeEnqueueTool

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    tool = AgentRuntimeEnqueueTool(
        requester=DaemonAgentRuntimeJobRequester.from_profiles(
            BUILTIN_INVOCATION_PROFILES,
            enabled=True,
        ),
        runtime=runtime,
    )

    result = await tool.execute(
        {
            "kind": "web_research",
            "profile_id": "web_research.readonly",
            "user_request": "Проверить sources.",
            "source_signal_id": "signal-tool",
            "chat_id": 100,
            "owner_user_id": 200,
        }
    )

    assert result.success is True
    assert result.data is not None
    job = await runtime.status(result.data["job_id"])
    assert job.status is AgentJobStatus.QUEUED
    assert job.profile.id == "web_research.readonly"
    assert "Проверить sources." not in result.message


def test_topic_signal_becomes_bounded_daemon_runtime_request() -> None:
    from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
    from src.agent_runtime.topic_signals import TopicClusterReadySignal
    from src.daemon.agent_runtime_requester import (
        DaemonAgentRuntimeJobRequester,
        build_daemon_request_from_topic_signal,
    )

    signal = TopicClusterReadySignal(
        cluster_key="codex-hooks",
        title="Codex hooks update",
        summary="Official docs describe lifecycle gates.",
        final_priority=91.0,
        recommended_route="spec",
        tier=2,
        payload={"source_url_0": "https://developers.openai.com/codex/hooks"},
    )

    request = build_daemon_request_from_topic_signal(
        signal,
        owner_user_id=200,
        chat_id=100,
    )
    plan = DaemonAgentRuntimeJobRequester.from_profiles(
        BUILTIN_INVOCATION_PROFILES,
        enabled=True,
    ).plan(request)

    assert request.kind == "topic_signal.spec"
    assert request.profile_id == "self_coding.readonly_discussion"
    assert request.source_signal_id == "topic_cluster_ready:codex-hooks"
    assert request.fingerprint == "topic_signal:codex-hooks:spec"
    assert request.metadata["topic_auto_execute_allowed"] == "false"
    assert request.metadata["topic_auto_publish_allowed"] == "false"
    assert request.metadata["topic_requires_approval"] == "true"
    assert plan.allowed is True
    assert plan.context_pack is not None
    assert "side_effects_require_separate_approval" in plan.context_pack.constraints
