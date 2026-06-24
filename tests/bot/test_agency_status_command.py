"""Read-only Agency Runtime status command wiring."""

from __future__ import annotations

from src.skills.base import AgentContext


async def test_agency_status_command_lists_agency_jobs_without_running_them() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.bot.main import _agency_status_reply

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=123,
        chat_id=123,
        source_message_id="tg:agency:1",
        fingerprint="agency-status",
        kind="agency",
        profile=InvocationProfile(id="agency.readonly_draft", worker="agency"),
        context_pack=ContextPack(user_request="Проверить agency status."),
    )
    await runtime.emit_progress(
        job.id,
        "Agency policy asks Никита before social side effects.",
    )

    status = await _agency_status_reply(
        "/agency_status",
        AgentContext(user_id=123, chat_id=123, mode="personal"),
        admin_user_id=123,
        runtime=runtime,
    )

    assert status is not None
    assert "Agency Runtime status:" in status
    assert "agency · agency.readonly_draft" in status
    assert "Agency policy asks Никита" in status
    assert "Проверить agency status." not in status


async def test_agency_status_command_is_admin_only() -> None:
    from src.bot.main import _agency_status_reply

    status = await _agency_status_reply(
        "/agency_status",
        AgentContext(user_id=456, chat_id=456, mode="assistant"),
        admin_user_id=123,
        runtime=None,
    )

    assert status == "Эта команда доступна только Никите."
