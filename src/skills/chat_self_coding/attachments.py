"""Attachment persistence for the /код self-coding room.

Telegram files are saved as raw artifacts in the personal workspace and
referenced from the chat self-coding discussion context. The code agent gets
the original path, not a lossy summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StoredAttachment:
    """A Telegram attachment persisted for later self-coding context."""

    kind: str
    path: Path
    workspace_path: str
    original_name: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class _AttachmentSpec:
    kind: str
    file_id: str
    filename: str
    content_type: str


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


async def save_message_attachments(
    messages: list[Any],
    *,
    workspace_root: Path,
    base_dir: str = "self_coding_uploads",
    now: datetime | None = None,
) -> tuple[StoredAttachment, ...]:
    """Download supported Telegram attachments into workspace storage."""
    timestamp = now or datetime.now(tz=UTC)
    upload_dir = workspace_root / base_dir / timestamp.strftime("%Y-%m-%d")
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: list[StoredAttachment] = []
    for message in messages:
        bot = getattr(message, "bot", None)
        if bot is None:
            continue
        spec = _attachment_spec_from_message(message)
        if spec is None:
            continue

        downloaded = await bot.download(spec.file_id)
        if downloaded is None:
            continue
        payload = downloaded.read()
        if not isinstance(payload, bytes) or not payload:
            continue

        index = len(saved)
        filename = _stored_filename(
            message_id=int(getattr(message, "message_id", 0)),
            index=index,
            kind=spec.kind,
            filename=spec.filename,
        )
        path = upload_dir / filename
        path.write_bytes(payload)
        saved.append(
            StoredAttachment(
                kind=spec.kind,
                path=path,
                workspace_path=path.relative_to(workspace_root).as_posix(),
                original_name=Path(spec.filename).name,
                content_type=spec.content_type,
                size_bytes=len(payload),
            )
        )
    return tuple(saved)


def format_attachment_context(
    attachments: tuple[StoredAttachment, ...],
    *,
    caption: str = "",
    target_label: str = "/код",
) -> str:
    """Render stored attachments as discussion context for Architect/Editor."""
    lines = [
        f"Никита прислал вложение для {target_label}. Это raw-контекст задачи: "
        "оригинал сохранён целиком, ниже только пути и метаданные.",
    ]
    if caption.strip():
        lines.append(f"Подпись Никиты: {caption.strip()}")
    for index, attachment in enumerate(attachments, start=1):
        lines.extend(
            [
                f"Вложение {index}:",
                f"- тип: {attachment.kind}",
                f"- имя: {attachment.original_name}",
                f"- content_type: {attachment.content_type or 'unknown'}",
                f"- размер: {attachment.size_bytes} bytes",
                f"- absolute_path: {attachment.path}",
                f"- workspace_path: {attachment.workspace_path}",
            ]
        )
    lines.append(
        "Architect/Editor: используй absolute_path как read-only источник; "
        "если это изображение, открывай оригинал напрямую."
    )
    return "\n".join(lines)


def _attachment_spec_from_message(message: Any) -> _AttachmentSpec | None:
    photo = getattr(message, "photo", None)
    if photo:
        largest = photo[-1]
        return _AttachmentSpec(
            kind="photo",
            file_id=str(largest.file_id),
            filename="photo.jpg",
            content_type="image/jpeg",
        )

    for kind, default_name, content_type in (
        ("document", "file.bin", "application/octet-stream"),
        ("video", "video.mp4", "video/mp4"),
        ("animation", "animation.gif", "image/gif"),
        ("audio", "audio.mp3", "audio/mpeg"),
        ("voice", "voice.ogg", "audio/ogg"),
        ("video_note", "video_note.mp4", "video/mp4"),
    ):
        item = getattr(message, kind, None)
        if item is None:
            continue
        return _AttachmentSpec(
            kind=kind,
            file_id=str(item.file_id),
            filename=str(getattr(item, "file_name", "") or default_name),
            content_type=str(getattr(item, "mime_type", "") or content_type),
        )
    return None


def _stored_filename(
    *,
    message_id: int,
    index: int,
    kind: str,
    filename: str,
) -> str:
    safe = _safe_filename(filename)
    return f"{message_id}_{index}_{kind}_{safe}"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name or "file.bin"
    safe = _SAFE_NAME_RE.sub("_", name.replace(" ", "_")).strip("._-")
    return safe or "file.bin"
