from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from src.skills.chat_self_coding.intent_classifier import Stage
from src.skills.chat_self_coding.state import ChatSelfCodingState


class _FakeStateStore:
    def __init__(self) -> None:
        self.state: ChatSelfCodingState | None = None

    async def load(self, user_id: int) -> ChatSelfCodingState | None:
        del user_id
        return self.state

    async def save(self, state: ChatSelfCodingState) -> None:
        self.state = state

    async def clear(self, user_id: int) -> None:
        del user_id
        self.state = None


class _FakeBot:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    async def download(self, file_id: str) -> BytesIO:
        return BytesIO(self.payloads[file_id])


def _message(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "message_id": 42,
        "from_user": SimpleNamespace(id=1),
        "chat": SimpleNamespace(id=1),
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


@pytest.fixture
def deps(tmp_path: Path) -> tuple[Any, _FakeStateStore]:
    from src.bot.handlers.self_coding_attachments import (
        reset_self_coding_attachment_deps_for_tests,
        set_self_coding_attachment_deps,
    )

    reset_self_coding_attachment_deps_for_tests()
    store = _FakeStateStore()
    set_self_coding_attachment_deps(
        admin_user_id=1,
        state_store=store,
        workspace_root=tmp_path,
    )
    return tmp_path, store


async def test_attachment_filter_matches_only_open_self_coding_session(
    deps: tuple[Any, _FakeStateStore],
) -> None:
    from src.bot.handlers.self_coding_attachments import SelfCodingAttachmentFilter

    _, store = deps
    msg = _message(photo=[SimpleNamespace(file_id="p")])
    filt = SelfCodingAttachmentFilter()

    assert await filt(msg, mode="personal") is False

    store.state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE, is_open=False)
    assert await filt(msg, mode="personal") is False

    store.state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE, is_open=True)
    assert await filt(msg, mode="personal") is True


async def test_document_attachment_is_saved_and_added_to_session_context(
    deps: tuple[Any, _FakeStateStore],
) -> None:
    from src.bot.handlers.self_coding_attachments import handle_self_coding_attachment

    tmp_path, store = deps
    store.state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
    msg = _message(
        bot=_FakeBot({"doc": b"important log"}),
        document=SimpleNamespace(
            file_id="doc",
            file_name="error.log",
            mime_type="text/plain",
        ),
        caption="лог падения",
    )

    await handle_self_coding_attachment(msg, mode="personal")

    assert store.state is not None
    [context_line] = store.state.recent_messages
    assert "Никита прислал вложение" in context_line
    assert "error.log" in context_line
    assert "лог падения" in context_line
    assert "raw" in context_line.lower()
    assert (tmp_path / "self_coding_uploads").exists()
    from src.dialogue.state import FileDialogueStateStore

    state = FileDialogueStateStore(tmp_path).load(1)
    assert state.last_intent == "self_coding_attachment"
    assert state.last_tool == "self_coding_attachments"
    assert state.last_result == "success"
    assert "лог падения" in state.last_user_message
    msg.answer.assert_awaited_once()
    assert "контекст /код" in msg.answer.await_args.args[0]


async def test_photo_attachment_saves_raw_file_without_vision_summary(
    deps: tuple[Any, _FakeStateStore],
) -> None:
    from src.bot.handlers import self_coding_attachments as handler

    _, store = deps
    store.state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
    msg = _message(
        bot=_FakeBot({"photo": b"image bytes"}),
        photo=[SimpleNamespace(file_id="photo")],
        caption="вот скрин",
    )

    await handler.handle_self_coding_attachment(msg, mode="personal")

    assert store.state is not None
    [context_line] = store.state.recent_messages
    assert "photo.jpg" in context_line
    assert "вот скрин" in context_line
    assert "оригинал" in context_line.lower()
