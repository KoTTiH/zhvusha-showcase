from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path
from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from src.core.config import get_settings
from src.skills.channel_writer.archive import save_published_post
from src.skills.channel_writer.media import (
    normalize_visual_metadata,
    send_approved_media,
    validate_approved_media,
)
from src.skills.channel_writer.outbox_posts import load_channel_post_text
from src.skills.workspace_session.workspace import get_workspace_path
from src.utils.telegram import send_long_message

logger = structlog.get_logger()

router = Router(name="workspace_session")

_outbox_root: Path | None = None


def set_outbox_root(path: Path) -> None:
    """Set outbox root for testing or manual override."""
    global _outbox_root
    _outbox_root = path


def _get_outbox_root() -> Path:
    if _outbox_root is not None:
        return _outbox_root
    settings = get_settings()
    ws = get_workspace_path(settings.workspace_path)
    return ws / "outbox"


def build_outbox_keyboard(item_type: str, filename: str) -> InlineKeyboardMarkup:
    """Build approve/skip keyboard for an outbox item."""
    approve_data = f"ws:approve:{item_type}:{filename}"
    skip_data = f"ws:skip:{item_type}:{filename}"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="\u2705 \u041e\u0434\u043e\u0431\u0440\u0438\u0442\u044c",
                    callback_data=approve_data,
                ),
                InlineKeyboardButton(
                    text="\u274c \u041f\u0440\u043e\u043f\u0443\u0441\u0442\u0438\u0442\u044c",
                    callback_data=skip_data,
                ),
            ]
        ]
    )


def _parse_callback_data(data: str) -> tuple[str, str, str]:
    """Parse 'ws:{action}:{item_type}:{filename}' → (action, item_type, filename)."""
    parts = data.split(":", 3)
    return parts[1], parts[2], parts[3]


def _move_to_processed(outbox_root: Path, item_type: str, filename: str) -> None:
    """Move file from outbox/{item_type}/ to outbox/.processed/."""
    src = outbox_root / item_type / filename
    dest_dir = outbox_root / ".processed"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dest_dir / filename))


@router.callback_query(F.data.startswith("ws:approve:"))
async def handle_approve(callback: CallbackQuery) -> None:
    """Approve an outbox item — execute its action."""
    _, item_type, filename = _parse_callback_data(
        callback.data  # type: ignore[arg-type]
    )
    outbox_root = _get_outbox_root()
    file_path = outbox_root / item_type / filename

    if not file_path.exists():
        await callback.answer(
            "\u0424\u0430\u0439\u043b \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d",
            show_alert=True,
        )
        return

    content = file_path.read_text(encoding="utf-8")

    if item_type == "channel_posts":
        # Send to channel
        settings = get_settings()
        if callback.message is not None and callback.message.bot is not None:
            raw, body = load_channel_post_text(content)
            raw_visual = raw.get("visual")
            visual = normalize_visual_metadata(
                raw_visual if isinstance(raw_visual, dict) else None
            )
            ws_root = get_workspace_path(settings.workspace_path)
            media_check = validate_approved_media(
                visual,
                workspace_root=ws_root,
                allow_ready=True,
            )
            if not media_check.allowed:
                await callback.answer(
                    f"Публикация заблокирована: {media_check.reason}",
                    show_alert=True,
                )
                return
            media = await send_approved_media(
                callback.message.bot,
                chat_id=settings.channel_id,
                validation=media_check,
            )
            messages = await send_long_message(
                callback.message.bot,
                chat_id=settings.channel_id,
                text=body.strip(),
            )
            if media is not None:
                media["text_message_id"] = messages[0].message_id
                media["text_parts"] = len(messages)
            await save_published_post(
                workspace_root=ws_root,
                text=body.strip(),
                message_id=messages[0].message_id,
                visual=visual,
                media=media,
            )
        await callback.answer(
            "\u041e\u043f\u0443\u0431\u043b\u0438\u043a\u043e\u0432\u0430\u043d\u043e!"
        )

    elif item_type == "kwork_drafts":
        # Show as copyable text
        if callback.message is not None:
            await callback.message.edit_text(  # type: ignore[union-attr]
                text=f"\U0001f4cb \u0414\u043b\u044f \u043a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f:\n\n<blockquote expandable>{content}</blockquote>",
                parse_mode="HTML",
            )
        await callback.answer("\u0413\u043e\u0442\u043e\u0432\u043e!")

    else:
        await callback.answer("\u041e\u0434\u043e\u0431\u0440\u0435\u043d\u043e")

    _move_to_processed(outbox_root, item_type, filename)
    logger.info("outbox_approved", item_type=item_type, filename=filename)


@router.callback_query(F.data.startswith("ws:skip:"))
async def handle_skip(callback: CallbackQuery) -> None:
    """Skip an outbox item — archive without action."""
    _, item_type, filename = _parse_callback_data(
        callback.data  # type: ignore[arg-type]
    )
    outbox_root = _get_outbox_root()

    _move_to_processed(outbox_root, item_type, filename)

    await callback.answer("\u041f\u0440\u043e\u043f\u0443\u0449\u0435\u043d\u043e")
    if callback.message is not None:
        try:
            await callback.message.delete()  # type: ignore[union-attr]
        except Exception:
            logger.debug("skip_delete_failed", message_id=callback.message.message_id)

    logger.info("outbox_skipped", item_type=item_type, filename=filename)
