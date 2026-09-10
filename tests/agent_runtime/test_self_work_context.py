"""Self-work Context Capsule tests."""

from __future__ import annotations


def test_self_work_context_capsule_reports_gaps_without_secret_leaks() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.self_work_context import (
        SelfWorkContextCapsuleBuilder,
        SelfWorkContextSnapshot,
        SelfWorkMcpHealth,
        SelfWorkRuntimeSignal,
    )

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_profile.telegram_mcp.personal_readonly",
                label="telegram_mcp.personal_readonly",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.DEGRADED,
                reason="SESSION=super_secret_session is missing from runtime",
                profile_id="telegram_mcp.personal_readonly",
            ),
            CapabilityNode(
                id="skill.unregistered_skill",
                label="unregistered_skill",
                kind=CapabilityKind.SKILL,
                status=CapabilityStatus.ORPHANED,
                reason="skill.yaml exists but startup routing does not register it",
            ),
        )
    )
    pending_job = AgentJob.new(
        owner_user_id=123,
        chat_id=123,
        source_message_id="cycle:1",
        fingerprint="fp",
        kind="self_improvement",
        profile=InvocationProfile(id="self_improvement.autonomous"),
        context_pack=ContextPack(user_request="cycle"),
    )

    capsule = SelfWorkContextCapsuleBuilder().build(
        SelfWorkContextSnapshot(
            capability_graph=graph,
            open_task_paths=(
                "tasks/2026-05-14-daemon-agent-runtime-job-requester.yaml",
            ),
            recent_failed_runs=("pytest failed: tests/runtime/test_gap.py::test_gap",),
            news_topic_backlog=("topic: runtime capability graph visibility",),
            mcp_health=(
                SelfWorkMcpHealth(
                    name="telegram-mcp-personal",
                    status=CapabilityStatus.DEGRADED,
                    reason="SESSION=super_secret_session unavailable",
                ),
            ),
            pending_jobs=(pending_job,),
            daemon_signals=(
                SelfWorkRuntimeSignal(
                    source="daemon",
                    signal_type="runtime_gap",
                    summary="daemon observed configured-only worker",
                    priority="normal",
                ),
            ),
        )
    )

    rendered = "\n".join(
        [capsule.summary, capsule.processed_context, capsule.markdown_report]
    )
    assert "agent_profile.telegram_mcp.personal_readonly" in rendered
    assert "skill.unregistered_skill" in rendered
    assert "super_secret_session" not in rendered
    assert "SESSION=" not in rendered
    assert any(
        finding.claim == "Capability gap: agent_profile.telegram_mcp.personal_readonly"
        for finding in capsule.findings
    )
    assert (
        "safe_spec_candidate:agent_profile.telegram_mcp.personal_readonly"
        in capsule.artifacts
    )
    assert any("bounded spec" in action for action in capsule.next_actions)


def test_self_work_context_capsule_filters_side_effect_candidates() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.self_work_context import (
        SelfWorkContextCapsuleBuilder,
        SelfWorkContextSnapshot,
    )

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.telegram_mcp.personal_actions.telegram_mcp_send",
                label="telegram_mcp_send",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.DEGRADED,
                reason="send tool unavailable",
                profile_id="telegram_mcp.personal_actions",
                capability_id="telegram_mcp_send",
            ),
        )
    )

    capsule = SelfWorkContextCapsuleBuilder().build(
        SelfWorkContextSnapshot(capability_graph=graph)
    )

    assert "safe_spec_candidate:" not in "\n".join(capsule.artifacts)
    assert any("side-effect" in action for action in capsule.next_actions)


def test_self_work_context_capsule_includes_topic_signals_without_execution() -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.agent_runtime.self_work_context import (
        SelfWorkContextCapsuleBuilder,
        SelfWorkContextSnapshot,
    )
    from src.agent_runtime.topic_signals import TopicClusterReadySignal

    signal = TopicClusterReadySignal(
        cluster_key="codex-hooks",
        title="OpenAI Codex hooks update",
        summary="Official docs describe new hooks for lifecycle gates.",
        final_priority=91.0,
        recommended_route="spec",
        tier=2,
        requires_approval=True,
        auto_publish_allowed=False,
        auto_execute_allowed=False,
        payload={"source_url_0": "https://developers.openai.com/codex/hooks"},
    )

    capsule = SelfWorkContextCapsuleBuilder().build(
        SelfWorkContextSnapshot(
            capability_graph=CapabilityGraph(capabilities=()),
            topic_signals=(signal,),
        )
    )
    rendered = "\n".join((capsule.processed_context, *capsule.artifacts))

    assert "codex-hooks" in rendered
    assert "recommended_route=spec" in rendered
    assert "auto_publish_allowed=False" in rendered
    assert "topic_signal:codex-hooks" in capsule.artifacts
    assert any(
        finding.claim == "Topic signal ready: codex-hooks"
        for finding in capsule.findings
    )
    assert any(
        "Create bounded spec candidate" in action for action in capsule.next_actions
    )
    assert not any(
        "publish" in action.lower() and "auto" in action.lower()
        for action in capsule.next_actions
    )
