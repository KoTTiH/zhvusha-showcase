"""Safe Telegram delivery helpers for user-facing bot responses."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog
from aiogram.exceptions import TelegramNetworkError

from src.utils.telegram import send_long_message

if TYPE_CHECKING:
    from aiogram import Bot

logger = structlog.get_logger(__name__)

_IMAGE_ARTIFACT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_IMAGE_ARTIFACT_REF_RE = re.compile(
    r"(?P<path>(?:/[^\s`<>\"']*/)?agent_runtime/"
    r"(?:computer_use/screenshots|browser_artifacts|channel_visual_artifacts)/"
    r"[^\s`<>\"']+\.(?:jpg|jpeg|png|webp))",
    re.IGNORECASE,
)


async def deliver_telegram_skill_response(
    *,
    bot: Bot,
    chat_id: int | str,
    text: str,
    parse_mode: str | None,
    artifacts: tuple[str, ...],
    workspace_root: Path,
) -> bool:
    """Deliver a skill response and persist payload if Telegram is unreachable."""

    image_artifacts = _resolve_image_artifacts(
        (*artifacts, *_image_artifact_refs_from_text(text)),
        workspace_root=workspace_root,
    )
    if not text and not image_artifacts:
        return True

    try:
        if text:
            await send_long_message(
                bot,
                chat_id,
                text,
                parse_mode=parse_mode,
            )
        if image_artifacts:
            from aiogram.types import FSInputFile

            for artifact in image_artifacts:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=FSInputFile(artifact),
                )
    except TelegramNetworkError as exc:
        failure_path = persist_telegram_delivery_failure(
            workspace_root=workspace_root,
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            artifacts=artifacts,
            image_artifacts=tuple(str(path) for path in image_artifacts),
            error=exc,
        )
        logger.warning(
            "telegram_skill_response_delivery_failed",
            chat_id=chat_id,
            error_type=type(exc).__name__,
            failure_path=str(failure_path) if failure_path is not None else "",
        )
        return False
    return True


def persist_telegram_delivery_failure(
    *,
    workspace_root: Path,
    chat_id: int | str,
    text: str,
    parse_mode: str | None,
    artifacts: tuple[str, ...],
    image_artifacts: tuple[str, ...],
    error: Exception,
) -> Path | None:
    """Persist one failed Telegram payload for audit and later manual retry."""

    root = workspace_root.expanduser().resolve()
    target_dir = root / "runtime" / "telegram_delivery_failures"
    payload = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "artifacts": list(artifacts),
        "image_artifacts": list(image_artifacts),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{_timestamp_slug()}-{uuid4().hex[:8]}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.exception(
            "telegram_delivery_failure_persist_failed",
            workspace_root=str(root),
        )
        return None
    return target


def _resolve_image_artifacts(
    artifacts: tuple[str, ...],
    *,
    workspace_root: Path,
) -> tuple[Path, ...]:
    root = workspace_root.expanduser().resolve()
    resolved: list[Path] = []
    for artifact_ref in artifacts:
        artifact = str(artifact_ref).strip()
        if not artifact:
            continue
        candidate = Path(artifact).expanduser()
        path = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not path.is_relative_to(root):
            logger.warning(
                "telegram_delivery_artifact_rejected",
                reason="path_escapes_workspace",
                artifact=artifact,
            )
            continue
        if path.suffix.lower() not in _IMAGE_ARTIFACT_SUFFIXES:
            continue
        if not path.is_file():
            logger.warning(
                "telegram_delivery_artifact_missing",
                artifact=artifact,
                path=str(path),
            )
            continue
        resolved.append(path)
    return tuple(dict.fromkeys(resolved))


def _image_artifact_refs_from_text(text: str) -> tuple[str, ...]:
    """Extract allowlisted local image artifact refs from a chat response."""
    refs: list[str] = []
    for match in _IMAGE_ARTIFACT_REF_RE.finditer(text):
        ref = match.group("path").rstrip(".,;:!?)\\]")
        if ref:
            refs.append(ref)
    return tuple(dict.fromkeys(refs))


def _timestamp_slug() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
