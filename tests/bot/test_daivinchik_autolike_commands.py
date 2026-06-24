from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from src.skills.base import AgentContext


class _BlockingDaivinchikWorker:
    name = "daivinchik_taste_profile"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.jobs: list[Any] = []
        self.stop_jobs: list[Any] = []
        self.cancelled: list[str] = []

    async def run(self, *, job: Any, context_pack: Any) -> Any:
        from src.agent_runtime.models import ContextCapsule

        request = json.loads(context_pack.user_request)
        if request.get("mode") == "autolike_stop":
            self.stop_jobs.append(job)
            return ContextCapsule(
                summary="Daivinchik stop-scrolling completed.",
                markdown_report="Daivinchik stop-scrolling completed.",
            )
        self.jobs.append(job)
        self.started.set()
        await self.release.wait()
        return ContextCapsule(
            summary="Daivinchik autolike live completed.",
            markdown_report="Daivinchik autolike live completed.\nactions=1",
        )

    async def cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        self.release.set()
        return True


async def _runtime(worker: Any) -> Any:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    return AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={worker.name: worker},
    )


async def test_daivinchik_start_command_creates_bounded_background_job() -> None:
    from src.agent_runtime.models import AgentJobStatus
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=99)

    with patch(
        "src.bot.main.get_settings",
        return_value=SimpleNamespace(daivinchik_chat_id="@leomatchbot"),
    ):
        reply = await _daivinchik_autolike_control_reply(
            "/daivinchik_start 7",
            context,
            admin_user_id=1,
            runtime=runtime,
        )

    assert reply is not None
    assert "Daivinchik autolike запущен" in reply
    assert "Статус:" not in reply
    await asyncio.wait_for(worker.started.wait(), timeout=1.0)
    jobs = await runtime.store.list_by_status((AgentJobStatus.RUNNING,))
    assert len(jobs) == 1
    job = jobs[0]
    request = json.loads(job.context_pack.user_request)
    assert request == {
        "chat_id": "@leomatchbot",
        "mode": "autolike_live",
        "attention_mode": "stop",
        "limit": 20,
        "max_actions": 7,
        "notify_chat_id": "1",
        "liked_forward_chat_id": "1",
    }
    assert job.profile.id == "telegram_mcp.daivinchik_autolike_bot_command"
    assert job.profile.allows("telegram_mcp_daivinchik_button")
    assert job.profile.allows("telegram_mcp_daivinchik_forward_liked_profile")
    assert job.profile.allows("telegram_mcp_daivinchik_notify")
    assert not job.profile.allows("telegram_mcp_send")
    assert (
        job.context_pack.metadata["agent_tool_approval_capabilities"]
        == "telegram_mcp_daivinchik_button,"
        "telegram_mcp_daivinchik_reply_button,"
        "telegram_mcp_daivinchik_notify,"
        "telegram_mcp_daivinchik_forward_liked_profile"
    )

    await runtime.cancel(job.id, reason="cleanup")
    await runtime.wait_background(job.id)


async def test_daivinchik_start_command_uses_default_chat_and_default_limit() -> None:
    from src.agent_runtime.models import AgentJobStatus
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=98)

    with patch(
        "src.bot.main.get_settings",
        return_value=SimpleNamespace(daivinchik_chat_id="@leomatchbot"),
    ):
        reply = await _daivinchik_autolike_control_reply(
            "/daivinchik_start",
            context,
            admin_user_id=1,
            runtime=runtime,
        )

    assert reply is not None
    assert "chat_id=@leomatchbot" in reply
    assert "max_actions=50" in reply
    await asyncio.wait_for(worker.started.wait(), timeout=1.0)
    jobs = await runtime.store.list_by_status((AgentJobStatus.RUNNING,))
    request = json.loads(jobs[0].context_pack.user_request)
    assert request["chat_id"] == "@leomatchbot"
    assert request["max_actions"] == 50

    await runtime.cancel(jobs[0].id, reason="cleanup")
    await runtime.wait_background(jobs[0].id)


async def test_daivinchik_start_rejects_invalid_numeric_limit() -> None:
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=97)

    with patch(
        "src.bot.main.get_settings",
        return_value=SimpleNamespace(daivinchik_chat_id="@leomatchbot"),
    ):
        reply = await _daivinchik_autolike_control_reply(
            "/daivinchik_start 0",
            context,
            admin_user_id=1,
            runtime=runtime,
        )

    assert reply is not None
    assert "Используй: /daivinchik_start [max_actions]" in reply
    assert worker.jobs == []


async def test_daivinchik_stop_command_cancels_active_live_job() -> None:
    from src.agent_runtime.models import AgentJobStatus
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=100)

    await _daivinchik_autolike_control_reply(
        "/daivinchik_start @leomatchbot 3",
        context,
        admin_user_id=1,
        runtime=runtime,
    )
    await asyncio.wait_for(worker.started.wait(), timeout=1.0)

    reply = await _daivinchik_autolike_control_reply(
        "/daivinchik_stop",
        context,
        admin_user_id=1,
        runtime=runtime,
    )

    assert reply is not None
    assert "остановлен" in reply
    jobs = await runtime.store.list_by_status((AgentJobStatus.CANCELED,))
    assert len(jobs) == 1
    assert worker.cancelled == [jobs[0].id]
    assert len(worker.stop_jobs) == 1
    stop_request = json.loads(worker.stop_jobs[0].context_pack.user_request)
    assert stop_request["mode"] == "autolike_stop"
    assert stop_request["chat_id"] == "@leomatchbot"


async def test_daivinchik_stop_command_runs_default_stop_pass_without_active_job() -> (
    None
):
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=103)

    with patch(
        "src.bot.main.get_settings",
        return_value=SimpleNamespace(daivinchik_chat_id="@leomatchbot"),
    ):
        reply = await _daivinchik_autolike_control_reply(
            "/daivinchik_stop",
            context,
            admin_user_id=1,
            runtime=runtime,
        )

    assert reply is not None
    assert "stop-pass выполнен" in reply
    assert len(worker.stop_jobs) == 1
    stop_request = json.loads(worker.stop_jobs[0].context_pack.user_request)
    assert stop_request["mode"] == "autolike_stop"
    assert stop_request["chat_id"] == "@leomatchbot"


async def test_daivinchik_commands_are_admin_only() -> None:
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)

    reply = await _daivinchik_autolike_control_reply(
        "/daivinchik_start @leomatchbot",
        AgentContext(user_id=2, chat_id=2, mode="assistant"),
        admin_user_id=1,
        runtime=runtime,
    )

    assert reply == "Эта команда доступна только Никите."


async def test_daivinchik_status_command_reports_active_job() -> None:
    from src.agent_runtime.models import AgentJobStatus
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=101)

    await _daivinchik_autolike_control_reply(
        "/daivinchik_start @leomatchbot 5",
        context,
        admin_user_id=1,
        runtime=runtime,
    )
    await asyncio.wait_for(worker.started.wait(), timeout=1.0)

    reply = await _daivinchik_autolike_control_reply(
        "/daivinchik_status",
        context,
        admin_user_id=1,
        runtime=runtime,
    )

    assert reply is not None
    assert "Daivinchik autolike status:" in reply
    assert "running" in reply
    assert "max_actions=5" in reply

    jobs = await runtime.store.list_by_status((AgentJobStatus.RUNNING,))
    await runtime.cancel(jobs[0].id, reason="cleanup")
    await runtime.wait_background(jobs[0].id)


async def test_daivinchik_start_command_notifies_control_chat_when_loop_stops() -> None:
    from src.agent_runtime.models import AgentJobStatus
    from src.bot.main import _daivinchik_autolike_control_reply

    worker = _BlockingDaivinchikWorker()
    runtime = await _runtime(worker)
    bot = AsyncMock()
    context = AgentContext(
        user_id=1,
        chat_id=2,
        mode="personal",
        message_id=102,
        bot=bot,
    )

    await _daivinchik_autolike_control_reply(
        "/daivinchik_start @leomatchbot 1",
        context,
        admin_user_id=1,
        runtime=runtime,
    )
    await asyncio.wait_for(worker.started.wait(), timeout=1.0)
    worker.release.set()
    jobs = await runtime.store.list_by_status((AgentJobStatus.RUNNING,))
    await runtime.wait_background(jobs[0].id)

    bot.send_message.assert_awaited()
    args, kwargs = bot.send_message.await_args
    assert args == ()
    assert kwargs["chat_id"] == 2
    assert "Daivinchik autolike завершился" in kwargs["text"]
    assert "@leomatchbot" in kwargs["text"]
