from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.bot import main as bot_main
from src.skills.chat_self_coding.events import BlockEvent, BlockEventType
from src.skills.chat_self_coding.intent_classifier import Stage
from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase


class FakeStateStore:
    def __init__(self, state: ChatSelfCodingState | None) -> None:
        self.state = state

    async def load(self, user_id: int) -> ChatSelfCodingState | None:
        del user_id
        return self.state

    async def save(self, state: ChatSelfCodingState) -> None:
        self.state = state

    async def clear(self, user_id: int) -> None:
        del user_id
        self.state = None


async def test_self_coding_progress_reuses_one_telegram_message() -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages: dict[str, int] = {}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.PREPARATION,
            slug="visual-pipeline",
            payload={},
        ),
        text="<b>prep</b>",
        progress_messages=progress_messages,
    )
    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="visual-pipeline",
            payload={},
        ),
        text="<b>implementation</b>",
        progress_messages=progress_messages,
    )
    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.DONE,
            slug="visual-pipeline",
            payload={},
        ),
        text="<b>done</b>",
        progress_messages=progress_messages,
    )

    bot.send_message.assert_awaited_once()
    assert bot.edit_message_text.await_count == 2
    assert bot.edit_message_text.await_args_list[0].kwargs["message_id"] == 42
    assert bot.edit_message_text.await_args_list[0].kwargs["text"] == (
        "<b>implementation</b>"
    )
    assert bot.edit_message_text.await_args_list[1].kwargs["text"] == "<b>done</b>"
    assert progress_messages == {}


async def test_self_coding_progress_is_keyed_by_code_task_id() -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages: dict[str, int] = {}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.PREPARATION,
            slug="old-slug",
            task_id="code-task-fixed",
            payload={},
        ),
        text="<b>prep</b>",
        progress_messages=progress_messages,
    )
    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="new-slug",
            task_id="code-task-fixed",
            payload={},
        ),
        text="<b>implementation</b>",
        progress_messages=progress_messages,
    )

    bot.send_message.assert_awaited_once()
    bot.edit_message_text.assert_awaited_once()
    assert progress_messages == {"code-task-fixed": 42}


async def test_self_coding_plan_stays_separate_from_progress_message() -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages: dict[str, int] = {}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.PLAN,
            slug="visual-pipeline",
            payload={},
        ),
        text="<b>plan</b>",
        progress_messages=progress_messages,
    )

    bot.send_message.assert_awaited_once()
    bot.edit_message_text.assert_not_awaited()
    assert progress_messages == {}


async def test_self_coding_human_progress_note_is_separate_from_status_bar() -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages = {"visual-pipeline": 42}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="visual-pipeline",
            payload={"message_kind": "note"},
        ),
        text="• Добавляю visual plan и safety gate.",
        progress_messages=progress_messages,
    )

    bot.send_message.assert_awaited_once()
    bot.edit_message_text.assert_not_awaited()
    assert progress_messages == {"visual-pipeline": 42}


async def test_render_human_progress_note_uses_codex_style_bullet() -> None:
    class Translator:
        async def translate(self, text: str, *, kind: object) -> str:
            del kind
            return text

    rendered = await bot_main._render_block_event(
        BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="visual-pipeline",
            payload={
                "message_kind": "note",
                "detail": "Добавляю visual plan и safety gate.",
            },
        ),
        Translator(),  # type: ignore[arg-type]
    )

    assert rendered == "• Добавляю visual plan и safety gate."


async def test_self_coding_status_heartbeat_refreshes_elapsed_time(
    monkeypatch,
) -> None:
    class Translator:
        async def translate(self, text: str, *, kind: object) -> str:
            del kind
            return text

    monkeypatch.setattr(bot_main, "_SELF_CODING_STATUS_HEARTBEAT_SECONDS", 0.01)
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages: dict[str, int] = {}
    heartbeat_tasks: dict[str, asyncio.Task[None]] = {}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="visual-pipeline",
            payload={
                "percent": 40,
                "detail": "Запускаю Codex Editor.",
                "stage": "запуск агента",
                "elapsed_seconds": 0,
            },
        ),
        text="<b>implementation</b>",
        progress_messages=progress_messages,
        heartbeat_tasks=heartbeat_tasks,
        translator=Translator(),  # type: ignore[arg-type]
    )

    try:
        await asyncio.sleep(0.03)
        assert bot.edit_message_text.await_count >= 1
        edited = bot.edit_message_text.await_args.kwargs["text"]
        assert "Codex Editor работает" in edited
        assert "Прошло:" in edited
        assert "40%" not in edited
    finally:
        for task in heartbeat_tasks.values():
            task.cancel()
        for task in heartbeat_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_terminal_error_without_existing_status_does_not_create_progress_anchor() -> (
    None
):
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
    )
    progress_messages: dict[str, int] = {}
    heartbeat_tasks: dict[str, asyncio.Task[None]] = {}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.ERROR,
            slug="visual-pipeline",
            task_id="code-task-fixed",
            payload={},
        ),
        text="<b>error</b>",
        progress_messages=progress_messages,
        heartbeat_tasks=heartbeat_tasks,
    )

    bot.send_message.assert_awaited_once()
    bot.edit_message_text.assert_not_awaited()
    assert progress_messages == {}
    assert heartbeat_tasks == {}


async def test_terminal_error_cancels_existing_progress_anchor() -> None:
    task = asyncio.create_task(asyncio.sleep(60))
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        edit_message_text=AsyncMock(),
    )
    progress_messages = {"code-task-fixed": 42}
    heartbeat_tasks = {"code-task-fixed": task}

    await bot_main._send_or_edit_block_message(
        bot=bot,
        chat_id=123,
        event=BlockEvent(
            user_id=123,
            event_type=BlockEventType.ERROR,
            slug="visual-pipeline",
            task_id="code-task-fixed",
            payload={},
        ),
        text="<b>error</b>",
        progress_messages=progress_messages,
        heartbeat_tasks=heartbeat_tasks,
    )

    bot.send_message.assert_not_awaited()
    bot.edit_message_text.assert_awaited_once()
    assert progress_messages == {}
    assert heartbeat_tasks == {}
    assert task.cancelled() or task.cancelling()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_block_event_maps_to_confirmed_task_phase() -> None:
    assert (
        bot_main._task_phase_from_block_event(
            BlockEvent(
                user_id=123,
                event_type=BlockEventType.IMPLEMENTATION,
                slug="visual-pipeline",
                payload={"stage": "проверки worktree"},
            )
        )
        is TaskPhase.COMMIT
    )
    assert (
        bot_main._task_phase_from_block_event(
            BlockEvent(
                user_id=123,
                event_type=BlockEventType.IMPLEMENTATION,
                slug="visual-pipeline",
                payload={"stage": "review gate"},
            )
        )
        is TaskPhase.REVIEW
    )
    assert (
        bot_main._task_phase_from_block_event(
            BlockEvent(
                user_id=123,
                event_type=BlockEventType.ERROR,
                slug="visual-pipeline",
                payload={},
            )
        )
        is TaskPhase.REPAIR
    )


async def test_block_event_updates_matching_code_task_phase() -> None:
    store = FakeStateStore(
        ChatSelfCodingState(
            user_id=123,
            stage=Stage.RUNNING,
            active_spec_slug="visual-pipeline",
            code_task_id="code-task-fixed",
            task_phase=TaskPhase.IMPLEMENTATION,
        )
    )

    await bot_main._update_code_task_phase_from_block_event(
        BlockEvent(
            user_id=123,
            event_type=BlockEventType.IMPLEMENTATION,
            slug="visual-pipeline",
            task_id="code-task-fixed",
            payload={"stage": "review gate"},
        ),
        store,
    )

    assert store.state is not None
    assert store.state.task_phase is TaskPhase.REVIEW


async def test_block_event_does_not_update_stale_code_task() -> None:
    store = FakeStateStore(
        ChatSelfCodingState(
            user_id=123,
            stage=Stage.RUNNING,
            active_spec_slug="visual-pipeline",
            code_task_id="code-task-current",
            task_phase=TaskPhase.IMPLEMENTATION,
        )
    )

    await bot_main._update_code_task_phase_from_block_event(
        BlockEvent(
            user_id=123,
            event_type=BlockEventType.DONE,
            slug="visual-pipeline",
            task_id="code-task-old",
            payload={},
        ),
        store,
    )

    assert store.state is not None
    assert store.state.task_phase is TaskPhase.IMPLEMENTATION
