"""Telegram/audit rendering tests for Agent Runtime outputs."""

from __future__ import annotations

from datetime import UTC, datetime


def test_context_capsule_chat_render_keeps_summary_findings_and_unknowns() -> None:
    from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
    from src.agent_runtime.rendering import render_capsule_for_chat

    capsule = ContextCapsule(
        summary="В проекте уже есть Explorer, но не runtime.",
        findings=(
            Finding(
                claim="Explorer подключён к /код.",
                status=FindingStatus.CONFIRMED,
                confidence=0.9,
                evidence=("src/skills/chat_self_coding/skill.py",),
            ),
            Finding(
                claim="Background jobs ещё не подключены.",
                status=FindingStatus.PARTIAL,
                confidence=0.6,
            ),
        ),
        next_actions=("Подключить durable AgentRuntime.",),
    )

    rendered = render_capsule_for_chat(capsule)

    assert "В проекте уже есть Explorer" in rendered
    assert "Explorer подключён" in rendered
    assert "Background jobs" in rendered
    assert "Подключить durable AgentRuntime" in rendered


def test_job_status_render_uses_curated_events_not_raw_cli_logs() -> None:
    from src.agent_runtime.models import (
        AgentEvent,
        AgentEventType,
        AgentJob,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.rendering import render_job_status

    job = AgentJob(
        id="job-1",
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    events = (
        AgentEvent(
            job_id="job-1",
            event_type=AgentEventType.PROGRESS,
            message="Сверяю post с self-coding flow.",
        ),
    )

    rendered = render_job_status(job, events)

    assert "source_compare" in rendered
    assert "queued" in rendered
    assert "Сверяю post" in rendered
    assert "Explored" not in rendered


def test_builtin_result_renderer_uses_capsule_for_readonly_profiles() -> None:
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        Finding,
        FindingStatus,
        InvocationProfile,
    )
    from src.agent_runtime.rendering import build_builtin_result_renderer_registry

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:render",
        fingerprint="render",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    capsule = ContextCapsule(
        summary="проверила",
        findings=(
            Finding(
                claim="source_compare вернул evidence",
                status=FindingStatus.CONFIRMED,
                confidence=0.8,
                evidence=("src/agent_runtime/rendering.py",),
            ),
        ),
        sources=("src/agent_runtime/rendering.py",),
        markdown_report="raw markdown should not hide evidence",
    )

    rendered = build_builtin_result_renderer_registry().render(job, capsule)

    assert "Что проверено" in rendered
    assert "source_compare вернул evidence" in rendered
    assert "raw markdown should not hide evidence" not in rendered


def test_builtin_result_renderer_preserves_self_coding_markdown() -> None:
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.rendering import build_builtin_result_renderer_registry

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:render-code",
        fingerprint="render-code",
        kind="self_coding",
        profile=InvocationProfile(id="self_coding.implementation"),
        context_pack=ContextPack(user_request="/spec_run x"),
    )
    capsule = ContextCapsule(
        summary="готово",
        markdown_report="✅ self-coding report",
    )

    rendered = build_builtin_result_renderer_registry().render(job, capsule)

    assert rendered == "✅ self-coding report"


def test_builtin_result_renderer_hides_personal_telegram_inbound_body() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.rendering import build_builtin_result_renderer_registry
    from src.agent_runtime.telegram_inbound import (
        PersonalTelegramInboundEvent,
        build_personal_telegram_inbound_capsule,
    )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:personal-inbound",
        fingerprint="personal-inbound",
        kind="personal_telegram_inbound",
        profile=InvocationProfile(id="personal_telegram.inbound_readonly"),
        context_pack=ContextPack(user_request="personal Telegram inbound event"),
    )
    capsule = build_personal_telegram_inbound_capsule(
        PersonalTelegramInboundEvent(
            event_id="tg-personal:render",
            chat_id="@devchat",
            text="личная фраза: token=12345",
            received_at=datetime(2026, 5, 14, tzinfo=UTC),
        )
    )

    rendered = build_builtin_result_renderer_registry().render(job, capsule)

    assert "can_auto_reply:false" in rendered
    assert "не отвечать автоматически" in rendered
    assert "личная фраза" not in rendered
    assert "token=12345" not in rendered
