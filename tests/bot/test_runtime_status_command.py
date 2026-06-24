"""Read-only Agent Runtime status command wiring."""

from __future__ import annotations

from src.skills.base import AgentContext


async def test_runtime_status_command_lists_recent_jobs_and_daemon_source() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.bot.main import _runtime_status_reply

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=123,
        chat_id=123,
        source_message_id="daemon:signal-1",
        fingerprint="daemon-job",
        kind="web_research",
        profile=InvocationProfile(id="web_research.readonly", worker="web_research"),
        context_pack=ContextPack(
            user_request="Проверить приватный daemon request.",
            metadata={"daemon_source_signal_id": "signal-1"},
        ),
    )
    await runtime.emit_progress(job.id, "Created by daemon requester.")

    status = await _runtime_status_reply(
        "/runtime_status",
        AgentContext(user_id=123, chat_id=123, mode="personal"),
        admin_user_id=123,
        runtime=runtime,
    )

    assert status is not None
    assert "Agent Runtime status:" in status
    assert "web_research · web_research.readonly" in status
    assert "daemon_source_signal_id: signal-1" in status
    assert "Created by daemon requester." in status
    assert "Проверить приватный daemon request." not in status


async def test_runtime_status_command_is_admin_only() -> None:
    from src.bot.main import _runtime_status_reply

    status = await _runtime_status_reply(
        "/runtime_status",
        AgentContext(user_id=456, chat_id=456, mode="assistant"),
        admin_user_id=123,
        runtime=None,
    )

    assert status == "Эта команда доступна только Никите."
