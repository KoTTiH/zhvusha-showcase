"""Production self-work context provider tests."""

from __future__ import annotations

from pathlib import Path


async def test_runtime_self_work_provider_collects_graph_tasks_and_pending_jobs(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.autonomous_self_coding.context_provider import (
        RuntimeSelfWorkContextProvider,
    )

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "2026-05-14-open.yaml").write_text(
        "slug: open\nstatus: in_progress\n", encoding="utf-8"
    )
    (tasks_dir / "2026-05-14-done.yaml").write_text(
        "slug: done\nstatus: done\n", encoding="utf-8"
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=123,
        chat_id=123,
        source_message_id="cycle",
        fingerprint="pending",
        kind="self_improvement",
        profile=InvocationProfile(id="self_improvement.autonomous"),
        context_pack=ContextPack(user_request="pending job"),
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_profile.telegram_mcp.personal_readonly",
                label="telegram_mcp",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.CONFIGURED_ONLY,
                reason="session_string=super_secret_session",
                profile_id="telegram_mcp.personal_readonly",
            ),
        )
    )
    provider = RuntimeSelfWorkContextProvider(
        capability_graph=graph,
        tasks_dir=tasks_dir,
        runtime=runtime,
    )

    capsule = await provider.build_self_work_context_capsule()

    assert "agent_profile.telegram_mcp.personal_readonly" in capsule.processed_context
    assert "tasks/2026-05-14-open.yaml" in capsule.processed_context
    assert "tasks/2026-05-14-done.yaml" not in capsule.processed_context
    assert job.id in capsule.processed_context
    assert "super_secret_session" not in capsule.processed_context
    assert "<redacted-secret>" in capsule.processed_context


async def test_runtime_self_work_provider_collects_recent_failed_self_coding_runs(
    tmp_path: Path,
) -> None:
    import json

    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.autonomous_self_coding.context_provider import (
        RuntimeSelfWorkContextProvider,
    )

    archive_dir = tmp_path / "self_coding_summaries" / "agent_runtime"
    archive_dir.mkdir(parents=True)
    (archive_dir / "20260514T120000Z-failed.json").write_text(
        json.dumps(
            {
                "slug": "failed-runtime-spec",
                "status": "failed",
                "summary": "Implementation stopped before verification.",
                "quality_warnings": ["missing_tests"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (archive_dir / "20260514T130000Z-ok.json").write_text(
        json.dumps(
            {
                "slug": "done-spec",
                "status": "completed",
                "summary": "Done.",
                "quality_warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    provider = RuntimeSelfWorkContextProvider(
        capability_graph=CapabilityGraph(capabilities=()),
        tasks_dir=tasks_dir,
    )

    capsule = await provider.build_self_work_context_capsule()

    assert "Recent failed runs" in capsule.processed_context
    assert "failed-runtime-spec" in capsule.processed_context
    assert "missing_tests" in capsule.processed_context
    assert "done-spec" not in capsule.processed_context


async def test_runtime_self_work_provider_collects_topic_provider_signal(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.spec_command.parser import SourceProvenance
    from src.skills.topic_to_spec.models import TopicRecord

    class TopicProvider:
        async def get_topic(self, key: str | None = None) -> TopicRecord | None:
            del key
            return TopicRecord(
                cluster_key="codex-hooks",
                title="Codex hooks update",
                summary="Official docs mention hooks for agent lifecycle gates.",
                top_terms=("codex", "hooks"),
                final_priority=91.0,
                source_provenance=(
                    SourceProvenance(
                        url="https://developers.openai.com/codex/hooks",
                        source_type="official_docs",
                        trust_tier="primary",
                        claim="Hooks provide lifecycle gates.",
                    ),
                ),
            )

    from src.skills.autonomous_self_coding.context_provider import (
        RuntimeSelfWorkContextProvider,
    )

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    provider = RuntimeSelfWorkContextProvider(
        capability_graph=CapabilityGraph(capabilities=()),
        tasks_dir=tasks_dir,
        topic_provider=TopicProvider(),
    )

    capsule = await provider.build_self_work_context_capsule()

    assert "Topic signals" in capsule.processed_context
    assert "codex-hooks" in capsule.processed_context
    assert "topic_signal:codex-hooks" in capsule.artifacts
    assert "https://developers.openai.com/codex/hooks" in capsule.findings[0].evidence


async def test_runtime_self_work_provider_collects_daemon_signal_provider(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.agent_runtime.self_work_context import SelfWorkRuntimeSignal
    from src.skills.autonomous_self_coding.context_provider import (
        RuntimeSelfWorkContextProvider,
    )

    class DaemonSignals:
        async def recent_signals(self) -> tuple[SelfWorkRuntimeSignal, ...]:
            return (
                SelfWorkRuntimeSignal(
                    source="daemon_audit",
                    signal_type="blocked_action",
                    summary="Need follow-up; token=super_secret_value",
                    priority="normal",
                ),
            )

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    provider = RuntimeSelfWorkContextProvider(
        capability_graph=CapabilityGraph(capabilities=()),
        tasks_dir=tasks_dir,
        daemon_signal_provider=DaemonSignals(),
    )

    capsule = await provider.build_self_work_context_capsule()

    assert "Daemon/runtime signals" in capsule.processed_context
    assert "daemon_audit/blocked_action/normal" in capsule.processed_context
    assert "super_secret_value" not in capsule.processed_context
    assert "<redacted-secret>" in capsule.processed_context


async def test_background_skill_accepts_provider_after_construction() -> None:
    from datetime import UTC, datetime
    from typing import Any

    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextCapsule
    from src.agent_runtime.profiles import SELF_IMPROVEMENT_AUTONOMOUS
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.autonomous_self_coding.skill import AutonomousSelfCodingSkill

    class Worker:
        name = "self_improvement"

        async def run(self, *, job: Any, context_pack: Any) -> ContextCapsule:
            del context_pack
            return ContextCapsule(summary="cycle done", artifacts=(f"job:{job.id}",))

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    class Provider:
        async def build_self_work_context_capsule(self) -> ContextCapsule:
            return ContextCapsule(
                summary="provider attached",
                processed_context="Capability gap: runtime",
            )

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_improvement": Worker()},
    )
    skill = AutonomousSelfCodingSkill(
        admin_user_id=123,
        runtime=runtime,
        profile=SELF_IMPROVEMENT_AUTONOMOUS,
        clock=lambda: datetime(2026, 5, 12, 10, 0, tzinfo=UTC),
    )

    skill.set_self_work_context_provider(Provider())
    result = await skill.run_once(cycle_id="cycle-provider")

    job = await runtime.status(result.metadata["agent_job_id"])
    assert job.context_pack.metadata["self_work_context_capsule"] == "true"
    assert "Capability gap: runtime" in job.context_pack.active_code_state
