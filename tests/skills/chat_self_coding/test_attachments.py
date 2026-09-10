from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _FakeBot:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.downloaded: list[str] = []

    async def download(self, file_id: str) -> BytesIO:
        self.downloaded.append(file_id)
        return BytesIO(self.payloads[file_id])


def _message(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "message_id": 42,
        "photo": None,
        "document": None,
        "video": None,
        "animation": None,
        "audio": None,
        "voice": None,
        "video_note": None,
        "caption": "",
        "bot": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


async def test_save_document_attachment_to_workspace(tmp_path: Path) -> None:
    from src.skills.chat_self_coding.attachments import save_message_attachments

    bot = _FakeBot({"doc-1": b"traceback text"})
    msg = _message(
        bot=bot,
        document=SimpleNamespace(
            file_id="doc-1",
            file_name="../bad name.log",
            mime_type="text/plain",
        ),
    )

    attachments = await save_message_attachments(
        [msg],
        workspace_root=tmp_path,
        now=datetime(2026, 5, 8, tzinfo=UTC),
    )

    assert len(attachments) == 1
    saved = attachments[0]
    assert saved.kind == "document"
    assert saved.path.read_bytes() == b"traceback text"
    assert (
        saved.workspace_path
        == "self_coding_uploads/2026-05-08/42_0_document_bad_name.log"
    )
    assert saved.original_name == "bad name.log"
    assert bot.downloaded == ["doc-1"]


async def test_save_photo_uses_largest_photo(tmp_path: Path) -> None:
    from src.skills.chat_self_coding.attachments import save_message_attachments

    bot = _FakeBot({"small": b"small", "large": b"large"})
    msg = _message(
        bot=bot,
        photo=[
            SimpleNamespace(file_id="small"),
            SimpleNamespace(file_id="large"),
        ],
    )

    attachments = await save_message_attachments(
        [msg],
        workspace_root=tmp_path,
        now=datetime(2026, 5, 8, tzinfo=UTC),
    )

    assert len(attachments) == 1
    assert attachments[0].kind == "photo"
    assert attachments[0].path.read_bytes() == b"large"
    assert attachments[0].workspace_path.endswith("42_0_photo_photo.jpg")
    assert bot.downloaded == ["large"]


def test_format_attachment_context_keeps_raw_path_without_replacing_file(
    tmp_path: Path,
) -> None:
    from src.skills.chat_self_coding.attachments import (
        StoredAttachment,
        format_attachment_context,
    )

    path = tmp_path / "self_coding_uploads" / "2026-05-08" / "42_0_photo.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"image")

    text = format_attachment_context(
        (
            StoredAttachment(
                kind="photo",
                path=path,
                workspace_path="self_coding_uploads/2026-05-08/42_0_photo.jpg",
                original_name="photo.jpg",
                content_type="image/jpeg",
                size_bytes=5,
            ),
        ),
        caption="скрин ошибки",
    )

    assert str(path) in text
    assert "self_coding_uploads/2026-05-08/42_0_photo.jpg" in text
    assert "скрин ошибки" in text
    assert "raw" in text.lower()
    assert "оригинал" in text.lower()
