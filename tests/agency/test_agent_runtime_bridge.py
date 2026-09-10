from __future__ import annotations

from src.agency.agent_runtime_bridge import build_agency_context_pack
from src.agency.models import (
    AgencyAction,
    AgencyActionKind,
    AgencyDataNeed,
    AgencyIntent,
    AgencyIntentKind,
    AgencyOutcomeKind,
)
from src.agent_runtime.models import (
    AgentJob,
    AgentJobStatus,
    ContextCapsule,
    InvocationProfile,
)
from src.agent_runtime.workers.agency import AgencyWorkerBackend


def _intent() -> AgencyIntent:
    return AgencyIntent(
        kind=AgencyIntentKind.SELF_COMPLEXIFICATION,
        source="test",
        goal="Разобраться, какой инструмент нужен для развития функции",
        why_complexification="Это закрывает пробел в функции.",
        data_needs=(AgencyDataNeed.FACTS,),
        expected_outcomes=(AgencyOutcomeKind.CONTEXT_CAPSULE,),
        candidate_actions=(
            AgencyAction(
                kind=AgencyActionKind.WEB_RESEARCH,
                capability="web_search_sources",
                description="Почитать источники.",
            ),
        ),
        evidence=("unit-test",),
    )


async def test_agency_worker_returns_context_capsule_with_next_actions() -> None:
    pack = build_agency_context_pack(_intent())
    worker = AgencyWorkerBackend()
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=1,
        source_message_id="1",
        fingerprint="agency-test",
        kind="agency",
        profile=InvocationProfile(id="agency.readonly_draft", worker="agency"),
        context_pack=pack,
        status=AgentJobStatus.QUEUED,
    )
    capsule = await worker.run(job=job, context_pack=pack)

    assert isinstance(capsule, ContextCapsule)
    assert "AgencyIntent" in capsule.summary
    assert capsule.next_actions
    assert capsule.memory_candidates
