"""Contract tests for the shared Agent Runtime value models."""

from __future__ import annotations

from datetime import UTC, datetime


def test_context_capsule_requires_structured_and_markdown_surfaces() -> None:
    from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus

    capsule = ContextCapsule(
        summary="В обычном чате нет полноценного codebase explorer.",
        processed_context="Проверены chat tools и /код Explorer path.",
        findings=(
            Finding(
                claim="Chat tools ограничены KB/workspace.",
                status=FindingStatus.CONFIRMED,
                confidence=0.91,
                evidence=("src/skills/chat_response/tools.py",),
            ),
        ),
        sources=("src/skills/chat_response/tools.py",),
        artifacts=("agent_jobs/job-1/report.md",),
        memory_candidates=("Обычный чат не должен обещать live repo audit.",),
        next_actions=("Запустить source_compare profile.",),
        markdown_report="## Итог\nОбычный чат не равен repo explorer.",
    )

    assert capsule.findings[0].status is FindingStatus.CONFIRMED
    assert capsule.markdown_report.startswith("## Итог")
    assert "source_compare" in capsule.next_actions[0]


def test_agent_job_keeps_owner_source_and_lifecycle_state() -> None:
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextPack,
        InvocationProfile,
    )

    job = AgentJob(
        id="job-1",
        owner_user_id=1291112109,
        chat_id=1291112109,
        source_message_id="telegram:42",
        fingerprint="abc",
        kind="source_compare",
        profile=InvocationProfile(
            id="source_compare.readonly",
            allowed_capabilities=("read_code", "read_attachments"),
            denied_capabilities=("write_files", "restart"),
        ),
        context_pack=ContextPack(
            user_request="Сравни пост с проектом.",
            chat_context=("Никита: ща скину пост",),
        ),
        status=AgentJobStatus.AWAITING_INPUT,
        created_at=datetime.now(UTC),
    )

    assert job.status is AgentJobStatus.AWAITING_INPUT
    assert job.profile.allows("read_code")
    assert not job.profile.allows("write_files")
    assert job.context_pack.chat_context == ("Никита: ща скину пост",)
