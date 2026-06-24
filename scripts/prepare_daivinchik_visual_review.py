#!/usr/bin/env python3
"""Prepare local visual review sheets for Daivinchik history.

This script is intentionally read/media-only:
- reads Telegram history;
- downloads profile media into a temporary review job directory;
- extracts representative video frames with ffmpeg;
- renders local HTML/PNG sheets where each card keeps text, media, and the
  visible reaction together.

It does not call LLM/Vision APIs and does not send/modify Telegram messages.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent_runtime.workers.daivinchik_profile import (
    ACTION_NEGATIVE,
    ACTION_POSITIVE,
    ACTION_UNKNOWN,
    _chronological_messages,
    _downloaded_path,
    _extract_age,
    _extract_city,
    _extract_messages,
    _infer_media_kind,
    _is_empty_media_placeholder,
    _looks_like_profile_text,
    _message_id,
    _message_media_refs,
    _message_text,
    _normalize_media_kind,
    _short_hash,
    _visible_action,
)
from src.agent_runtime.workers.telegram_mcp import MCPStdioTelegramClient
from src.core.config import get_settings
from src.skills.workspace_session.workspace import get_workspace_path

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_HISTORY_LIMIT = 10_000
DEFAULT_BATCH_SIZE = 8
SHEET_WIDTH = 1800
SHEET_HEIGHT = 3600
READ_TOOLS = frozenset({"get_history", "get_media_info", "download_media"})


@dataclass
class ReviewMedia:
    message_id: str
    media_id: str
    media_hash: str
    declared_kind: str
    kind: str = "unknown"
    info_hash: str = ""
    downloaded_path: Path | None = None
    review_paths: list[Path] = field(default_factory=list)
    error: str = ""


@dataclass
class ReviewCard:
    index: int
    message_ids: list[str]
    raw_text: str
    action: str
    media: list[ReviewMedia]
    age: int | None
    city: str
    content_hash: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local Daivinchik text+media+reaction review sheets."
    )
    parser.add_argument("chat_id", help="Exact Daivinchik chat id or username.")
    parser.add_argument("--limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--job-id", default="")
    parser.add_argument(
        "--skip-screenshots",
        action="store_true",
        help="Only write HTML and manifest; do not run Chromium screenshots.",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.telegram_mcp_enabled:
        sys.stderr.write("TELEGRAM_MCP_ENABLED=true is required.\n")
        return 2
    workspace_root = get_workspace_path(settings.workspace_path)
    job_id = (
        args.job_id.strip() or f"job-{int(time.time())}-{_short_hash(args.chat_id)}"
    )
    job_root = (
        workspace_root
        / "telegram-mcp"
        / "daivinchik-visual-review"
        / _safe_path_part(job_id)
    )
    if job_root.exists():
        shutil.rmtree(job_root)
    media_root = job_root / "media"
    frame_root = job_root / "frames"
    html_root = job_root / "html"
    sheet_root = job_root / "sheets"
    for path in (media_root, frame_root, html_root, sheet_root):
        path.mkdir(parents=True, exist_ok=True)

    async with MCPStdioTelegramClient(read_timeout_seconds=120.0) as client:
        raw_history = await _call_read_tool(
            client,
            "get_history",
            {"chat_id": args.chat_id, "limit": max(int(args.limit), 1)},
        )
        cards = normalize_review_cards(raw_history)
        await _download_card_media(
            client=client,
            chat_id=str(args.chat_id),
            cards=cards,
            media_root=media_root,
            frame_root=frame_root,
        )

    _write_cards_jsonl(job_root / "cards_review.jsonl", cards)
    _write_public_manifest(job_root / "manifest.json", args.chat_id, cards)
    html_files = _write_review_html(
        html_root, cards, batch_size=max(args.batch_size, 1)
    )
    sheet_files: list[Path] = []
    if not args.skip_screenshots:
        sheet_files = await _render_sheets(html_files, sheet_root)
    _write_sheet_index(job_root / "sheet_index.md", html_files, sheet_files)
    print(f"job_root: {job_root}")
    print(f"cards: {len(cards)}")
    print(f"media_refs: {sum(len(card.media) for card in cards)}")
    print(
        f"media_errors: {sum(1 for card in cards for media in card.media if media.error)}"
    )
    print(f"html_batches: {len(html_files)}")
    print(f"sheets: {len(sheet_files)}")
    return 0


def normalize_review_cards(raw_history: Any) -> list[ReviewCard]:
    """Group history into cards: text + media bundle + reaction below."""
    messages = _chronological_messages(_extract_messages(raw_history))
    cards: list[ReviewCard] = []
    current_texts: list[str] = []
    current_message_ids: list[str] = []
    current_media: list[ReviewMedia] = []

    def flush(action: str = ACTION_UNKNOWN) -> None:
        nonlocal current_texts, current_message_ids, current_media
        if not current_texts and not current_media:
            return
        raw_text = "\n".join(current_texts).strip()
        basis = "\n".join(
            [
                re.sub(r"\s+", " ", raw_text.casefold()).strip(),
                *[media.media_hash for media in current_media],
                action,
            ]
        )
        cards.append(
            ReviewCard(
                index=len(cards) + 1,
                message_ids=list(current_message_ids),
                raw_text=raw_text,
                action=action or ACTION_UNKNOWN,
                media=list(current_media),
                age=_extract_age(raw_text),
                city=_extract_city(raw_text),
                content_hash=_short_hash(basis),
            )
        )
        current_texts = []
        current_message_ids = []
        current_media = []

    for message in messages:
        text = _message_text(message)
        media_refs = _message_media_refs(message)
        action = _visible_action(text)
        if action != ACTION_UNKNOWN:
            flush(action)
            continue
        if text and _looks_like_profile_text(text) and (current_texts or current_media):
            flush(ACTION_UNKNOWN)
        if text:
            current_texts.append(text)
            current_message_ids.append(_message_id(message))
        elif _is_empty_media_placeholder(message):
            current_message_ids.append(_message_id(message))
        for media_ref in media_refs:
            current_media.append(
                ReviewMedia(
                    message_id=media_ref.message_id,
                    media_id=media_ref.media_id,
                    media_hash=media_ref.media_hash,
                    declared_kind=media_ref.kind,
                )
            )
    flush(ACTION_UNKNOWN)
    return cards


async def _download_card_media(
    *,
    client: MCPStdioTelegramClient,
    chat_id: str,
    cards: list[ReviewCard],
    media_root: Path,
    frame_root: Path,
) -> None:
    for card in cards:
        for media_index, media in enumerate(card.media, start=1):
            stem = f"card-{card.index:04d}-msg-{media.message_id}-{media.media_hash}"
            try:
                info_text = await _call_read_tool(
                    client,
                    "get_media_info",
                    {"chat_id": chat_id, "message_id": media.message_id},
                )
                media.info_hash = _short_hash(str(info_text))
                media.kind = _infer_media_kind(media.declared_kind, str(info_text))
                downloaded = await _call_read_tool(
                    client,
                    "download_media",
                    {
                        "chat_id": chat_id,
                        "message_id": media.message_id,
                        "file_path": str(media_root / stem),
                    },
                )
                media.downloaded_path = _downloaded_path(
                    str(downloaded), temp_dir=media_root
                )
                if media.kind == "unknown":
                    media.kind = _kind_from_path(media.downloaded_path)
                if media.kind == "video":
                    media.review_paths = await _extract_video_frames(
                        media.downloaded_path,
                        frame_root=frame_root,
                        stem=f"{stem}-media-{media_index}",
                    )
                else:
                    media.review_paths = [media.downloaded_path]
            except Exception as exc:
                media.error = f"{type(exc).__name__}: {exc}"


async def _call_read_tool(
    client: MCPStdioTelegramClient,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    if tool_name not in READ_TOOLS:
        raise ValueError(f"tool is not read/media-read allowed: {tool_name}")
    return await client.call_tool(tool_name, arguments)


async def _extract_video_frames(
    video_path: Path,
    *,
    frame_root: Path,
    stem: str,
) -> list[Path]:
    duration = await _probe_duration(video_path)
    if duration > 4:
        offsets = [0.8, duration * 0.5, max(duration - 1.0, 0.8)]
    elif duration > 0:
        offsets = [min(0.5, duration * 0.25), duration * 0.5]
    else:
        offsets = [0.0]
    frames: list[Path] = []
    for index, offset in enumerate(offsets, start=1):
        frame_path = frame_root / f"{stem}-frame-{index}.jpg"
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{offset:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode == 0 and frame_path.exists():
            frames.append(frame_path)
        elif not frames:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or "ffmpeg failed to extract video frame")
    return frames


async def _probe_duration(path: Path) -> float:
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await process.communicate()
    if process.returncode != 0:
        return 0.0
    try:
        return max(float(stdout.decode("utf-8", errors="replace").strip()), 0.0)
    except ValueError:
        return 0.0


def _write_cards_jsonl(path: Path, cards: list[ReviewCard]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(
                json.dumps(_card_to_json(card, include_text=True), ensure_ascii=False)
            )
            handle.write("\n")


def _write_public_manifest(path: Path, chat_id: str, cards: list[ReviewCard]) -> None:
    path.write_text(
        json.dumps(
            {
                "chat_id": chat_id,
                "cards": len(cards),
                "positive": sum(1 for card in cards if card.action == ACTION_POSITIVE),
                "negative": sum(1 for card in cards if card.action == ACTION_NEGATIVE),
                "unknown": sum(1 for card in cards if card.action == ACTION_UNKNOWN),
                "media_refs": sum(len(card.media) for card in cards),
                "media_errors": sum(
                    1 for card in cards for media in card.media if media.error
                ),
                "cards_with_media": sum(1 for card in cards if card.media),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _card_to_json(card: ReviewCard, *, include_text: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": card.index,
        "content_hash": card.content_hash,
        "message_ids": card.message_ids,
        "action": card.action,
        "age": card.age,
        "city": card.city,
        "media": [
            {
                "message_id": media.message_id,
                "media_hash": media.media_hash,
                "kind": media.kind,
                "declared_kind": media.declared_kind,
                "info_hash": media.info_hash,
                "downloaded_path": str(media.downloaded_path)
                if media.downloaded_path
                else "",
                "review_paths": [str(path) for path in media.review_paths],
                "error": media.error,
            }
            for media in card.media
        ],
    }
    if include_text:
        payload["text"] = card.raw_text
    else:
        payload["text_hash"] = _short_hash(card.raw_text)
    return payload


def _write_review_html(
    html_root: Path,
    cards: list[ReviewCard],
    *,
    batch_size: int,
) -> list[Path]:
    files: list[Path] = []
    for batch_index, start in enumerate(range(0, len(cards), batch_size), start=1):
        batch = cards[start : start + batch_size]
        path = html_root / f"batch-{batch_index:03d}.html"
        path.write_text(_render_html_batch(batch_index, batch), encoding="utf-8")
        files.append(path)
    return files


def _render_html_batch(batch_index: int, cards: list[ReviewCard]) -> str:
    cards_html = "\n".join(_render_card_html(card) for card in cards)
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Daivinchik review batch {batch_index}</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 24px;
  width: {SHEET_WIDTH}px;
  background: #101214;
  color: #f1f3f4;
  font-family: Inter, Arial, sans-serif;
}}
.sheet-title {{
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 18px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}}
.card {{
  min-height: 820px;
  border: 2px solid #34383d;
  background: #181b1f;
  padding: 16px;
  overflow: hidden;
}}
.card.like {{ border-color: #37b26c; }}
.card.skip {{ border-color: #df5b57; }}
.card.unknown {{ border-color: #80868b; }}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 10px;
  font-size: 20px;
  font-weight: 700;
}}
.badge {{
  padding: 4px 9px;
  border-radius: 4px;
  background: #2b3036;
}}
.like .reaction {{ background: #1d6b42; }}
.skip .reaction {{ background: #812f2c; }}
.unknown .reaction {{ background: #4b5158; }}
.text {{
  min-height: 130px;
  max-height: 185px;
  white-space: pre-wrap;
  overflow: hidden;
  color: #d6d9dc;
  font-size: 18px;
  line-height: 1.25;
  border-left: 4px solid #5f6872;
  padding-left: 10px;
  margin-bottom: 12px;
}}
.media-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}}
.media-grid.single {{ grid-template-columns: 1fr; }}
.media {{
  background: #050607;
  min-height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}}
.media img {{
  max-width: 100%;
  max-height: 250px;
  object-fit: contain;
}}
.media-label {{
  position: absolute;
  top: 6px;
  left: 6px;
  padding: 3px 6px;
  background: rgba(0,0,0,.72);
  font-size: 14px;
}}
.error {{
  color: #ffb4ab;
  font-size: 15px;
  white-space: pre-wrap;
}}
</style>
</head>
<body>
<h1 class="sheet-title">Daivinchik review batch {batch_index}</h1>
<div class="grid">
{cards_html}
</div>
</body>
</html>
"""


def _render_card_html(card: ReviewCard) -> str:
    cls = {
        ACTION_POSITIVE: "like",
        ACTION_NEGATIVE: "skip",
    }.get(card.action, "unknown")
    text = html.escape(card.raw_text[:900] or "(нет текста)")
    media_items = _review_media_items(card)
    media_cls = "single" if len(media_items) == 1 else ""
    media_html = "\n".join(media_items) or '<div class="error">нет media</div>'
    age_city = " / ".join(
        item for item in (str(card.age) if card.age else "", card.city) if item
    )
    return f"""<section class="card {cls}">
  <div class="meta">
    <span class="badge">#{card.index}</span>
    <span class="badge reaction">{html.escape(_action_label(card.action))}</span>
    <span class="badge">{html.escape(age_city or "age/city unknown")}</span>
    <span class="badge">{html.escape(card.content_hash)}</span>
  </div>
  <div class="text">{text}</div>
  <div class="media-grid {media_cls}">
    {media_html}
  </div>
</section>"""


def _review_media_items(card: ReviewCard) -> list[str]:
    items: list[str] = []
    for media_index, media in enumerate(card.media, start=1):
        if media.error:
            items.append(
                '<div class="media"><div class="media-label">'
                f"{media_index} {html.escape(media.kind)}</div>"
                f'<div class="error">{html.escape(media.error[:400])}</div></div>'
            )
            continue
        for frame_index, path in enumerate(media.review_paths, start=1):
            label = f"{media_index}.{frame_index} {media.kind}"
            items.append(
                '<div class="media">'
                f'<div class="media-label">{html.escape(label)}</div>'
                f'<img src="{path.as_uri()}" alt="{html.escape(label)}">'
                "</div>"
            )
    return items


async def _render_sheets(html_files: list[Path], sheet_root: Path) -> list[Path]:
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        raise RuntimeError("chromium is required for screenshots")
    sheets: list[Path] = []
    for html_file in html_files:
        output = sheet_root / f"{html_file.stem}.png"
        process = await asyncio.create_subprocess_exec(
            chromium,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--screenshot={output}",
            f"--window-size={SHEET_WIDTH},{SHEET_HEIGHT}",
            html_file.as_uri(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"chromium screenshot failed for {html_file}: {detail}")
        sheets.append(output)
    return sheets


def _write_sheet_index(
    path: Path,
    html_files: list[Path],
    sheet_files: list[Path],
) -> None:
    lines = ["# Daivinchik Visual Review Sheets", ""]
    for index, html_file in enumerate(html_files, start=1):
        sheet = sheet_files[index - 1] if index <= len(sheet_files) else None
        lines.append(
            f"- batch {index:03d}: html={html_file}"
            + (f" sheet={sheet}" if sheet else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kind_from_path(path: Path) -> str:
    suffix = path.suffix.casefold()
    return _normalize_media_kind(suffix)


def _action_label(action: str) -> str:
    return {
        ACTION_POSITIVE: "LIKE",
        ACTION_NEGATIVE: "SKIP",
        ACTION_UNKNOWN: "UNKNOWN",
    }.get(action, action.upper())


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:80] or "job"


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
