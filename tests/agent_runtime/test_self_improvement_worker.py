"""Agent Runtime worker tests for autonomous self-improvement."""

from __future__ import annotations


def test_self_improvement_profile_is_capability_scoped() -> None:
    from src.agent_runtime.profiles import (
        SELF_IMPROVEMENT_AUTONOMOUS,
        build_builtin_agent_registry,
        build_builtin_capability_registry,
    )

    agents = build_builtin_agent_registry()
    capabilities = build_builtin_capability_registry()

    assert agents.get("self_improvement").default_worker == "self_improvement"
    assert SELF_IMPROVEMENT_AUTONOMOUS.worker == "self_improvement"
    assert "write_files" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "restart" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "publish" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "browser_submit" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "send_message" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "edit_env" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "commit" in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities
    assert "write_whitelisted_files_after_approval" in (
        SELF_IMPROVEMENT_AUTONOMOUS.allowed_capabilities
    )
    assert "request_tier3_specs_for_nikita_approval" in (
        SELF_IMPROVEMENT_AUTONOMOUS.allowed_capabilities
    )
    capabilities.validate_profile(SELF_IMPROVEMENT_AUTONOMOUS)


async def test_self_improvement_worker_returns_context_capsule() -> None:
    from src.agent_runtime.models import (
        AgentJob,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.workers.self_improvement import (
        AutonomousSelfCodingWorkerBackend,
        SelfImprovementCycleResult,
    )

    calls: list[tuple[str, str]] = []

    class Engine:
        async def run_once(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> SelfImprovementCycleResult:
            calls.append((job.id, context_pack.user_request))
            return SelfImprovementCycleResult(
                status="started_implementation",
                summary="Started safe autonomous spec.",
                spec_slug="safe-autonomous-spec",
                implementation_job_id="job-impl",
                change_summary_path="/workspace/self_coding_summaries/safe.md",
                memory_candidates=("Safe self-work cycle succeeded.",),
                next_actions=("Wait for implementation completion.",),
            )

    worker = AutonomousSelfCodingWorkerBackend(engine=Engine())
    job = AgentJob.new(
        owner_user_id=123,
        chat_id=123,
        source_message_id="cycle:1",
        fingerprint="fp",
        kind="self_improvement",
        profile=InvocationProfile(id="self_improvement.autonomous"),
        context_pack=ContextPack(user_request="find one safe improvement"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert calls == [(job.id, "find one safe improvement")]
    assert capsule.summary == "Started safe autonomous spec."
    assert "safe-autonomous-spec" in capsule.artifacts
    assert "job-impl" in capsule.artifacts
    assert "change_summary_path: /workspace/self_coding_summaries/safe.md" in (
        capsule.artifacts
    )
    assert capsule.memory_candidates == ("Safe self-work cycle succeeded.",)
    assert capsule.next_actions == ("Wait for implementation completion.",)


async def test_self_improvement_worker_reports_engine_skip() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_improvement import (
        AutonomousSelfCodingWorkerBackend,
        SelfImprovementCycleResult,
    )

    class Engine:
        async def run_once(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> SelfImprovementCycleResult:
            del job, context_pack
            return SelfImprovementCycleResult(
                status="skipped",
                summary="Skipped because another self-coding job is active.",
            )

    worker = AutonomousSelfCodingWorkerBackend(engine=Engine())
    job = AgentJob.new(
        owner_user_id=123,
        chat_id=123,
        source_message_id="cycle:1",
        fingerprint="fp",
        kind="self_improvement",
        profile=InvocationProfile(id="self_improvement.autonomous"),
        context_pack=ContextPack(user_request="find one safe improvement"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.summary.startswith("Skipped")
    assert capsule.findings[0].claim == "Autonomous self-improvement status: skipped"
