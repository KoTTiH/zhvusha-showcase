from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from src.bot import main as bot_main
from src.llm.protocols import LLMResponse, LLMUsage
from src.skills.base import SkillResult


class _BlockingSkill:
    name = "chat_response"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    async def can_handle(self, message: str, context: Any) -> float:
        del message, context
        return 1.0

    async def execute(self, message: str, context: Any) -> SkillResult:
        del context
        self.calls.append(message)
        if message == "первый вопрос":
            self.started.set()
            await self.release.wait()
        return SkillResult(success=True, response=f"ответ: {message}")


def _message(text: str, *, message_id: int, bot: Any) -> Any:
    return SimpleNamespace(
        text=text,
        from_user=SimpleNamespace(id=12345),
        chat=SimpleNamespace(id=12345),
        message_id=message_id,
        bot=bot,
        answer=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _reset_busy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    bot_main._reset_chat_busy_state_for_tests()
    monkeypatch.setattr(bot_main, "_agent_runtime", None)
    monkeypatch.setattr(bot_main, "_source_compare_background_runner", None)
    router = SimpleNamespace(
        generate=AsyncMock(
            return_value=LLMResponse(
                text="new_topic",
                model="worker",
                usage=LLMUsage(),
            )
        )
    )
    monkeypatch.setattr(bot_main, "get_router", lambda: router)


async def test_busy_status_question_replies_without_new_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    status = _message("ты зависла?", message_id=2, bot=bot)
    await bot_main.handle_text(status)

    status.answer.assert_not_awaited()
    assert skill.calls == ["первый вопрос"]

    skill.release.set()
    await first_task


async def test_busy_self_coding_progress_question_is_status_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    status = _message("она до сих пор код пишет?", message_id=2, bot=bot)
    await bot_main.handle_text(status)

    status.answer.assert_not_awaited()
    assert skill.calls == ["первый вопрос"]

    skill.release.set()
    await first_task


async def test_busy_runtime_status_command_bypasses_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()
    monkeypatch.setattr(
        bot_main,
        "get_settings",
        lambda: SimpleNamespace(
            admin_user_id=12345,
            workspace_path=str(tmp_path / "ws"),
            project_path=str(tmp_path / "project"),
        ),
    )

    status = _message("/runtime_status", message_id=2, bot=bot)
    try:
        await bot_main.handle_text(status)
    finally:
        skill.release.set()
        await first_task

    status.answer.assert_awaited_once()
    assert "Agent Runtime status" in status.answer.await_args.args[0]
    assert skill.calls == ["первый вопрос"]


async def test_busy_new_topic_is_queued_and_drained_after_first_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    queued = _message("новая тема", message_id=2, bot=bot)
    await bot_main.handle_text(queued)

    queued.answer.assert_not_awaited()
    assert skill.calls == ["первый вопрос"]

    skill.release.set()
    await first_task

    assert skill.calls == ["первый вопрос", "новая тема"]
    sent_texts = [call.kwargs["text"] for call in bot.send_message.await_args_list]
    assert "ответ: первый вопрос" in sent_texts
    assert "ответ: новая тема" in sent_texts


async def test_stale_active_state_does_not_defer_first_new_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    async with bot_main._CHAT_BUSY_LOCK:
        bot_main._active_response_by_chat[12345] = bot_main._ChatBusyState(active=True)

    first = _message("новая первая тема", message_id=1, bot=bot)
    await bot_main.handle_text(first)

    first.answer.assert_not_awaited()
    assert skill.calls == ["новая первая тема"]
    sent_texts = [call.kwargs["text"] for call in bot.send_message.await_args_list]
    assert "ответ: новая первая тема" in sent_texts


async def test_busy_addendum_is_drained_before_regular_queued_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    queued = _message("новая тема", message_id=2, bot=bot)
    addendum = _message("+ учти вот это", message_id=3, bot=bot)
    await bot_main.handle_text(queued)
    await bot_main.handle_text(addendum)

    queued.answer.assert_not_awaited()
    addendum.answer.assert_not_awaited()

    skill.release.set()
    await first_task

    assert skill.calls == ["первый вопрос", "+ учти вот это", "новая тема"]


async def test_busy_queue_has_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    queued_messages = [
        _message(f"новая тема {idx}", message_id=idx + 2, bot=bot) for idx in range(6)
    ]
    for msg in queued_messages:
        await bot_main.handle_text(msg)

    assert "Очередь ответов заполнена" in queued_messages[-1].answer.await_args.args[0]
    skill.release.set()
    await first_task

    assert skill.calls == [
        "первый вопрос",
        "новая тема 0",
        "новая тема 1",
        "новая тема 2",
        "новая тема 3",
        "новая тема 4",
    ]


async def test_busy_ambiguous_status_uses_worker_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    router = SimpleNamespace(
        generate=AsyncMock(
            return_value=LLMResponse(text="status", model="worker", usage=LLMUsage())
        )
    )
    monkeypatch.setattr(bot_main, "get_router", lambda: router)
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    status = _message("ты там жива вообще?", message_id=2, bot=bot)
    await bot_main.handle_text(status)

    status.answer.assert_not_awaited()
    assert skill.calls == ["первый вопрос"]
    router.generate.assert_awaited_once()

    skill.release.set()
    await first_task


async def test_busy_ambiguous_addendum_uses_worker_classifier_and_priority_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    router = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                LLMResponse(text="new_topic", model="worker", usage=LLMUsage()),
                LLMResponse(text="addendum", model="worker", usage=LLMUsage()),
            ]
        )
    )
    monkeypatch.setattr(bot_main, "get_router", lambda: router)
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    queued = _message("новая тема", message_id=2, bot=bot)
    addendum = _message("сюда же: проверь тон", message_id=3, bot=bot)
    await bot_main.handle_text(queued)
    await bot_main.handle_text(addendum)

    queued.answer.assert_not_awaited()
    addendum.answer.assert_not_awaited()

    skill.release.set()
    await first_task

    assert skill.calls == ["первый вопрос", "сюда же: проверь тон", "новая тема"]


async def test_busy_classifier_failure_falls_back_to_new_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    router = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(bot_main, "get_router", lambda: router)
    bot = SimpleNamespace(send_message=AsyncMock())

    first = _message("первый вопрос", message_id=1, bot=bot)
    first_task = asyncio.create_task(bot_main.handle_text(first))
    await skill.started.wait()

    queued = _message("нетипичная фраза", message_id=2, bot=bot)
    await bot_main.handle_text(queued)

    queued.answer.assert_not_awaited()
    skill.release.set()
    await first_task

    assert skill.calls == ["первый вопрос", "нетипичная фраза"]


async def test_owner_pending_updates_replay_sequentially() -> None:
    active = 0
    max_active = 0
    seen: list[int] = []

    async def feed_update(_bot: Any, update: Any) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        seen.append(update)
        await asyncio.sleep(0)
        active -= 1

    dp = SimpleNamespace(feed_update=AsyncMock(side_effect=feed_update))
    bot = SimpleNamespace()

    await bot_main._replay_owner_pending_updates(dp, bot, [1, 2, 3])

    assert seen == [1, 2, 3]
    assert max_active == 1


async def test_active_agent_job_status_bypasses_skill_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="tg:agent",
        fingerprint="agent",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))
    await runtime.emit_progress(job.id, "Сверяю пост с кодовой базой.")
    monkeypatch.setattr(bot_main, "_agent_runtime", runtime)
    bot = SimpleNamespace(send_message=AsyncMock())

    status = _message("что там, зависла?", message_id=1, bot=bot)
    await bot_main.handle_text(status)

    status.answer.assert_awaited_once()
    assert "Сверяю пост с кодовой базой" in status.answer.await_args.args[0]
    assert skill.calls == []


async def test_active_agent_job_followup_is_attached_before_chat_busy_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="tg:agent",
        fingerprint="agent",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))
    monkeypatch.setattr(bot_main, "_agent_runtime", runtime)
    bot = SimpleNamespace(send_message=AsyncMock())

    followup = _message("+ вот ещё лог", message_id=1, bot=bot)
    await bot_main.handle_text(followup)

    updated = await runtime.status(job.id)
    followup.answer.assert_awaited_once()
    assert "добавила" in followup.answer.await_args.args[0]
    assert updated.followups == ("+ вот ещё лог",)
    assert skill.calls == []


async def test_active_agent_job_new_topic_continues_normal_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="tg:agent",
        fingerprint="agent",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))
    monkeypatch.setattr(bot_main, "_agent_runtime", runtime)
    bot = SimpleNamespace(send_message=AsyncMock())

    new_topic = _message("новая тема: придумай пост", message_id=1, bot=bot)
    await bot_main.handle_text(new_topic)

    updated = await runtime.status(job.id)
    new_topic.answer.assert_not_awaited()
    assert updated.followups == ()
    assert skill.calls == ["новая тема: придумай пост"]
    bot.send_message.assert_awaited_once()
    assert (
        bot.send_message.await_args.kwargs["text"] == "ответ: новая тема: придумай пост"
    )


async def test_awaiting_source_compare_starts_after_material_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    skill = _BlockingSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    job = await runtime.create_job(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="tg:awaiting",
        fingerprint="awaiting",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни с постом"),
        status=AgentJobStatus.AWAITING_INPUT,
    )

    class Runner:
        def __init__(self) -> None:
            self.started: list[str] = []

        async def start_existing_background(self, **kwargs: Any) -> object:
            self.started.append(kwargs["job_id"])
            await kwargs["completion_callback"]("готовый отчёт")
            return object()

    runner = Runner()
    monkeypatch.setattr(bot_main, "_agent_runtime", runtime)
    monkeypatch.setattr(bot_main, "_source_compare_background_runner", runner)
    bot = SimpleNamespace(send_message=AsyncMock())

    material = _message(
        "Anthropic представил Dreaming для агентов", message_id=1, bot=bot
    )
    await bot_main.handle_text(material)

    updated = await runtime.status(job.id)
    material.answer.assert_awaited_once()
    assert "Запустила agent-задачу" in material.answer.await_args.args[0]
    assert runner.started == [job.id]
    assert updated.followups == ("Anthropic представил Dreaming для агентов",)
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["text"] == "готовый отчёт"
    assert skill.calls == []
