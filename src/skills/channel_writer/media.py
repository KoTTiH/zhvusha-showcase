"""Approved media sink for channel writer drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.skills.post_drafts.visual_assets import resolve_workspace_asset
from src.skills.post_drafts.visual_plan import is_public_source_url

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram import Bot

TELEGRAM_PHOTO_CAPTION_MAX_LENGTH = 1024


@dataclass(frozen=True)
class MediaValidation:
    allowed: bool
    reason: str = ""
    should_publish: bool = False
    asset_path: Path | None = None
    asset_path_ref: str = ""
    caption: str = ""
    source_url: str = ""


def validate_approved_media(
    visual: dict[str, Any] | None,
    *,
    workspace_root: Path,
    allow_ready: bool = False,
) -> MediaValidation:
    """Validate optional approved media without deciding visual intent."""
    normalized = normalize_visual_metadata(visual)
    if not normalized:
        return MediaValidation(allowed=True)

    intent = str(normalized.get("intent", "none"))
    required = bool(normalized.get("required", False))
    status = str(normalized.get("status", ""))
    if intent in {"none", ""}:
        return MediaValidation(allowed=True)
    if intent == "denied":
        return MediaValidation(allowed=False, reason="visual denied by safety plan")
    if intent.startswith("source"):
        source_url = str(normalized.get("source_url", "")).strip()
        if not source_url:
            return MediaValidation(
                allowed=False,
                reason="source visual is missing source_url",
            )
        if not is_public_source_url(source_url):
            return MediaValidation(
                allowed=False,
                reason="source_url is not a public source",
            )
    else:
        source_url = ""
    approved_status = status == "approved" or (allow_ready and status == "ready")
    if not approved_status:
        if required:
            return MediaValidation(
                allowed=False,
                reason="visual-required draft has no approved asset",
            )
        return MediaValidation(allowed=True)
    try:
        asset = resolve_workspace_asset(
            workspace_root=workspace_root,
            asset_path=str(normalized.get("asset_path", "")),
        )
    except (FileNotFoundError, ValueError) as exc:
        return MediaValidation(
            allowed=False,
            reason=f"approved visual asset is outside workspace or missing: {exc}",
        )
    return MediaValidation(
        allowed=True,
        should_publish=True,
        asset_path=asset,
        asset_path_ref=asset.relative_to(
            workspace_root.expanduser().resolve()
        ).as_posix(),
        caption=str(normalized.get("caption", ""))[:1024],
        source_url=source_url,
    )


def normalize_visual_metadata(visual: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize canonical and legacy visual frontmatter into one contract."""
    if not visual:
        return None
    normalized = dict(visual)

    if "asset_path" not in normalized and "artifact_path" in normalized:
        normalized["asset_path"] = normalized["artifact_path"]

    if "intent" in normalized:
        return normalized

    needed = bool(normalized.get("needed", False))
    visual_type = str(normalized.get("type", "")).strip().lower()
    status = str(normalized.get("status", "") or "planned")
    if not needed and not visual_type:
        return None

    if visual_type in {"screenshot", "source_screenshot", "source", "source_image"}:
        intent = "source_screenshot"
    elif visual_type in {"generated", "technical_schema", "schema", "diagram"}:
        intent = "generated"
    else:
        intent = "generated" if needed else "none"

    normalized["intent"] = intent
    normalized["required"] = needed or intent not in {"none", ""}
    normalized["status"] = status
    if "caption" not in normalized:
        normalized["caption"] = (
            "Публичный источник к посту"
            if intent.startswith("source")
            else "Визуальная схема к мысли Жвуши"
        )
    if "safety_notes" not in normalized:
        safety = normalized.get("safety_notes") or normalized.get("safety")
        normalized["safety_notes"] = [str(safety)] if safety else []
    return normalized


def validate_media_caption_text(text: str) -> MediaValidation:
    """Require optional visual captions to fit Telegram's photo caption limit."""
    if len(text) <= TELEGRAM_PHOTO_CAPTION_MAX_LENGTH:
        return MediaValidation(allowed=True)
    return MediaValidation(
        allowed=False,
        reason=(
            "text is too long for a single Telegram photo caption "
            f"({len(text)}/{TELEGRAM_PHOTO_CAPTION_MAX_LENGTH})"
        ),
    )


async def send_approved_media(
    bot: Bot,
    *,
    chat_id: int | str,
    validation: MediaValidation,
) -> dict[str, Any] | None:
    """Send already approved media as its own post before the text post."""
    if not validation.should_publish or validation.asset_path is None:
        return None
    from aiogram.types import FSInputFile

    caption = validation.caption.rstrip()
    caption_check = validate_media_caption_text(caption)
    if not caption_check.allowed:
        raise ValueError(caption_check.reason)
    message = await bot.send_photo(
        chat_id=chat_id,
        photo=FSInputFile(validation.asset_path),
        caption=caption or None,
    )
    metadata = {
        "status": "published",
        "message_id": message.message_id,
        "asset_path": validation.asset_path_ref,
        "caption": validation.caption,
        "attached_to_text": False,
    }
    if validation.source_url:
        metadata["source_url"] = validation.source_url
    return metadata
