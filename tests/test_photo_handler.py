from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.bot.handlers.photo import handle_photo, set_photo_deps
from src.llm.protocols import LLMResponse, LLMUsage

if TYPE_CHECKING:
    from pathlib import Path


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="test", usage=LLMUsage())


@pytest.fixture
def ws_root(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "personality").mkdir(parents=True)
    (ws / "personality" / "core.md").write_text("I am Zhvusha.")
    (ws / "personality" / "genes.md").write_text("Curiosity: HIGH")
    (ws / "memory" / "people").mkdir(parents=True)
    (ws / "media").mkdir(parents=True)
    return ws


@pytest.fixture
def mock_router() -> AsyncMock:
    router = AsyncMock()
    router.describe_images = AsyncMock(return_value=_llm_resp("A cat on a couch"))
    router.generate = AsyncMock(return_value=_llm_resp("Какой милый котик! 🐱"))
    return router


@pytest.fixture
def mock_message(ws_root: Path) -> MagicMock:
    msg = MagicMock()
    msg.message_id = 42
    msg.from_user = MagicMock()
    msg.from_user.id = 12345
    msg.from_user.username = "nikita"
    msg.caption = "мой кот"
    msg.photo = [MagicMock(), MagicMock()]  # two sizes, [-1] is largest
    msg.photo[-1].file_id = "photo_file_id"

    bio = BytesIO(b"\xff\xd8\xff\xe0fake-jpeg-data")
    msg.bot = AsyncMock()
    msg.bot.download = AsyncMock(return_value=bio)

    msg.chat = MagicMock()
    msg.chat.id = 12345

    msg.answer = AsyncMock()
    return msg


@pytest.fixture(autouse=True)
def _setup(ws_root: Path) -> None:
    set_photo_deps(ws_root=ws_root, episodic=None)


async def test_photo_saves_to_media(
    ws_root: Path, mock_router: AsyncMock, mock_message: MagicMock
) -> None:
    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    media_files = list((ws_root / "media").iterdir())
    assert len(media_files) == 1
    assert media_files[0].name.endswith("_42_0.jpg")
    assert media_files[0].read_bytes() == b"\xff\xd8\xff\xe0fake-jpeg-data"


async def test_photo_calls_vision_then_text_llm(
    ws_root: Path, mock_router: AsyncMock, mock_message: MagicMock
) -> None:
    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    # Gemini was called for vision
    mock_router.describe_images.assert_awaited_once()
    vision_req = mock_router.describe_images.call_args.args[0]
    assert len(vision_req.images) == 1
    assert vision_req.images[0] == b"\xff\xd8\xff\xe0fake-jpeg-data"

    # Text LLM was called for response with Zhvusha's hard personality anchor.
    mock_router.generate.assert_awaited_once()
    gen_req = mock_router.generate.call_args.args[0]
    assert gen_req.tier == "analyst"
    assert "Непереписываемая личность" in gen_req.system

    # Response sent to user
    mock_message.answer.assert_awaited_once_with("Какой милый котик! 🐱")


async def test_photo_with_caption(
    ws_root: Path, mock_router: AsyncMock, mock_message: MagicMock
) -> None:
    mock_message.caption = "посмотри на это"
    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    # Vision prompt includes caption context
    vision_req = mock_router.describe_images.call_args.args[0]
    assert "посмотри на это" in vision_req.prompt

    # User prompt includes caption
    gen_req = mock_router.generate.call_args.args[0]
    assert "посмотри на это" in gen_req.prompt


async def test_photo_records_episode(
    ws_root: Path, mock_router: AsyncMock, mock_message: MagicMock
) -> None:
    mock_episodic = AsyncMock()
    mock_episodic.record = AsyncMock(return_value=1)
    set_photo_deps(ws_root=ws_root, episodic=mock_episodic)

    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    mock_episodic.record.assert_awaited_once()
    call_kwargs = mock_episodic.record.call_args.kwargs
    assert "photo_paths" in call_kwargs["metadata"]
    assert len(call_kwargs["metadata"]["photo_paths"]) == 1
    assert "A cat on a couch" in call_kwargs["content"]


async def test_photo_no_gemini_graceful(ws_root: Path, mock_message: MagicMock) -> None:
    """Without Gemini, describe_images returns a fallback message."""
    mock_router = AsyncMock()
    mock_router.describe_images = AsyncMock(
        return_value=_llm_resp("Не могу видеть фото, Gemini API не настроен")
    )
    mock_router.generate = AsyncMock(
        return_value=_llm_resp("Не вижу фото, но подпись прочитала!")
    )

    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    # Text LLM still generates a response.
    mock_router.generate.assert_awaited_once()
    mock_message.answer.assert_awaited_once()


async def test_photo_album(ws_root: Path, mock_router: AsyncMock) -> None:
    """Multiple photos in an album are all downloaded and described."""
    msg1 = MagicMock()
    msg1.message_id = 10
    msg1.from_user = MagicMock()
    msg1.from_user.id = 12345
    msg1.from_user.username = "nikita"
    msg1.caption = "альбом"
    msg1.photo = [MagicMock()]
    msg1.photo[-1].file_id = "file_1"
    msg1.bot = AsyncMock()
    msg1.bot.download = AsyncMock(return_value=BytesIO(b"img1"))
    msg1.chat = MagicMock()
    msg1.chat.id = 12345
    msg1.answer = AsyncMock()

    msg2 = MagicMock()
    msg2.message_id = 11
    msg2.photo = [MagicMock()]
    msg2.photo[-1].file_id = "file_2"
    msg2.bot = AsyncMock()
    msg2.bot.download = AsyncMock(return_value=BytesIO(b"img2"))

    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(msg1, album=[msg1, msg2])

    # Both images sent to Gemini
    vision_req = mock_router.describe_images.call_args.args[0]
    assert len(vision_req.images) == 2
    assert vision_req.images[0] == b"img1"
    assert vision_req.images[1] == b"img2"

    # Both saved to media/
    media_files = list((ws_root / "media").iterdir())
    assert len(media_files) == 2


async def test_photo_logs_to_chat_history(
    ws_root: Path, mock_router: AsyncMock, mock_message: MagicMock
) -> None:
    """Photo handler logs both the user photo and bot response to JSONL."""
    with (
        patch("src.bot.handlers.photo.get_router", return_value=mock_router),
        patch("src.bot.handlers.photo.get_people_manager") as mock_people,
    ):
        mock_people.return_value.get_profile_for_context.return_value = ""
        await handle_photo(mock_message)

    # Check log files
    import json

    log_files = list((ws_root / "logs").rglob("chat_*.jsonl"))
    assert len(log_files) == 1

    lines = log_files[0].read_text().strip().split("\n")
    assert len(lines) == 2  # user photo + bot response

    user_entry = json.loads(lines[0])
    assert user_entry["role"] == "user"
    assert user_entry["photo_paths"] is not None
    assert "photo_description" in user_entry
    assert user_entry["text"] == "мой кот"

    bot_entry = json.loads(lines[1])
    assert bot_entry["role"] == "assistant"
    assert bot_entry["text"] == "Какой милый котик! 🐱"
