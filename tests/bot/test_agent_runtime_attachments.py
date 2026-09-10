from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


class _FakeBot:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.sent: list[tuple[int, str]] = []

    async def download(self, file_id: str) -> BytesIO:
        return BytesIO(self.payloads[file_id])

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> None:
        del kwargs
        self.sent.append((chat_id, text))


class _FakeBackgroundRunner:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def start_existing_background(
        self,
        *,
        job_id: str,
        completion_callback: Any = None,
    ) -> None:
        del completion_callback
        self.started.append(job_id)


def _message(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "message_id": 42,
        "from_user": SimpleNamespace(id=1),
        "chat": SimpleNamespace(id=2),
        "photo": None,
        "document": None,
        "video": None,
        "animation": None,
        "audio": None,
        "voice": None,
        "video_note": None,
        "caption": "",
        "bot": _FakeBot({}),
        "answer": AsyncMock(),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


async def _runtime() -> Any:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore

    return AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )


@pytest.fixture(autouse=True)
def _reset_deps() -> None:
    from src.bot.handlers.agent_runtime_attachments import (
        reset_agent_runtime_attachment_deps_for_tests,
    )

    reset_agent_runtime_attachment_deps_for_tests()


async def test_filter_matches_only_when_active_agent_job_exists(tmp_path: Path) -> None:
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.bot.handlers.agent_runtime_attachments import (
        AgentRuntimeAttachmentFilter,
        set_agent_runtime_attachment_deps,
    )

    runtime = await _runtime()
    set_agent_runtime_attachment_deps(
        admin_user_id=1,
        workspace_root=tmp_path,
        runtime=runtime,
    )
    message = _message(photo=[SimpleNamespace(file_id="photo")])
    filt = AgentRuntimeAttachmentFilter()

    assert await filt(message, mode="personal") is False

    await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сейчас скину скрин"),
    )

    assert await filt(message, mode="personal") is True


async def test_attachment_to_awaiting_source_compare_starts_job(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import (
        AgentJobStatus,
        ContextPack,
        InvocationProfile,
    )
    from src.bot.handlers.agent_runtime_attachments import (
        handle_agent_runtime_attachment,
        set_agent_runtime_attachment_deps,
    )

    runtime = await _runtime()
    runner = _FakeBackgroundRunner()
    set_agent_runtime_attachment_deps(
        admin_user_id=1,
        workspace_root=tmp_path,
        runtime=runtime,
        source_compare_background_runner=runner,
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="ща скину пост скрином"),
        status=AgentJobStatus.AWAITING_INPUT,
    )
    message = _message(
        bot=_FakeBot({"photo": b"image bytes"}),
        photo=[SimpleNamespace(file_id="photo")],
        caption="вот скрин поста",
    )

    await handle_agent_runtime_attachment(message, mode="personal")

    updated = await runtime.status(job.id)
    assert runner.started == [job.id]
    assert updated.artifacts
    assert updated.artifacts[0].endswith("42_0_photo_photo.jpg")
    assert updated.followups
    assert "agent job" in updated.followups[0]
    assert "вот скрин поста" in updated.followups[0]
    assert (tmp_path / "agent_runtime_uploads").exists()
    from src.dialogue.state import FileDialogueStateStore

    state = FileDialogueStateStore(tmp_path).load(2)
    assert state.last_intent == "agent_runtime_attachment"
    assert state.last_tool == "agent_runtime_attachments"
    assert state.last_result == "success"
    assert "вот скрин поста" in state.last_user_message
    message.answer.assert_awaited_once()
    assert "Запустила agent-задачу" in message.answer.await_args.args[0]


async def test_attachment_to_running_job_is_added_without_start(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import (
        AgentJobStatus,
        ContextPack,
        InvocationProfile,
    )
    from src.bot.handlers.agent_runtime_attachments import (
        handle_agent_runtime_attachment,
        set_agent_runtime_attachment_deps,
    )

    runtime = await _runtime()
    runner = _FakeBackgroundRunner()
    set_agent_runtime_attachment_deps(
        admin_user_id=1,
        workspace_root=tmp_path,
        runtime=runtime,
        source_compare_background_runner=runner,
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )
    await runtime.store.save(job.with_status(AgentJobStatus.RUNNING))
    message = _message(
        bot=_FakeBot({"doc": b"log"}),
        document=SimpleNamespace(
            file_id="doc",
            file_name="error.log",
            mime_type="text/plain",
        ),
        caption="лог к анализу",
    )

    await handle_agent_runtime_attachment(message, mode="personal")

    updated = await runtime.status(job.id)
    assert runner.started == []
    assert updated.artifacts[0].endswith("42_0_document_error.log")
    assert "лог к анализу" in updated.followups[0]
    message.answer.assert_awaited_once()
    assert "добавила к текущей agent-задаче" in message.answer.await_args.args[0]
