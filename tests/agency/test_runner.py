from __future__ import annotations

from datetime import UTC, datetime

from src.personality.protocols import AffectiveSnapshot


def _snapshot() -> AffectiveSnapshot:
    return AffectiveSnapshot(
        self_emotion="curiosity",
        self_valence=0.4,
        self_arousal=0.85,
        user_emotion="neutral",
        user_valence=0.0,
        user_arousal=0.3,
        regulation_active=False,
        regulation_target="",
        turns_since_update=1,
        last_updated=datetime(2026, 5, 14, tzinfo=UTC),
    )


async def test_agency_runner_builds_policy_job_and_returns_context_capsule() -> None:
    from src.agency.intent_builder import PersonalityDrivenIntentBuilder
    from src.agency.models import AutonomyDecisionType
    from src.agency.policy import AutonomyPolicy
    from src.agency.runner import AgencyRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.workers.agency import AgencyWorkerBackend

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"agency": AgencyWorkerBackend()},
    )
    runner = AgencyRunner(
        builder=PersonalityDrivenIntentBuilder(),
        policy=AutonomyPolicy(),
        runtime=runtime,
    )

    result = await runner.run_once(
        owner_user_id=1,
        chat_id=100,
        source_message_id="tg:100:1",
        event="Разобраться, почему social grants не стали live flow",
        affective_snapshot=_snapshot(),
        desire_signals=("хочу получить человеческое мнение перед social action",),
    )

    assert result.policy_decision.decision is AutonomyDecisionType.ASK_NIKITA
    assert result.job.status is AgentJobStatus.DONE
    assert result.job.profile.id == "agency.readonly_draft"
    assert result.job.profile.worker == "agency"
    assert "send_message" in result.job.profile.denied_capabilities
    assert result.capsule is not None
    assert "AgencyIntent" in result.capsule.summary
    assert "AgencyIntent" in result.capsule.markdown_report
    assert "AutonomyPolicy: ask_nikita" in result.capsule.markdown_report
    assert "blocked: telegram_mcp_send" in result.capsule.markdown_report
    assert result.next_actions
    assert result.permission_request is not None
    assert result.permission_request.target_id
