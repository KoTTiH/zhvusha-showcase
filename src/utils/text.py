"""Text utilities — leaf module with no project dependencies."""

from __future__ import annotations

import re

_TELEGRAM_MAX_LENGTH = 4096


def md_to_tg_html(text: str) -> str:
    """Convert markdown formatting to Telegram-compatible HTML.

    Handles: **bold**, *italic*, `code`, ```code blocks```,
    ~~strikethrough~~. Escapes HTML special characters in plain text.
    """
    # First, extract code blocks and inline code to protect them
    placeholders: list[str] = []

    def _placeholder(match: re.Match[str]) -> str:
        idx = len(placeholders)
        placeholders.append(match.group(0))
        return f"\x00PH{idx}\x00"

    # Extract fenced code blocks (``` ... ```)
    text = re.sub(r"```(?:\w+)?\n(.*?)\n```", _placeholder, text, flags=re.DOTALL)
    text = re.sub(r"```(.*?)```", _placeholder, text, flags=re.DOTALL)

    # Extract inline code (`...`)
    text = re.sub(r"`([^`]+)`", _placeholder, text)

    # Escape HTML special characters in remaining text
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Bold+italic nested: ***text*** or **text *inner***
    # Process bold first, allowing content to include single *
    text = re.sub(r"\*\*(.+?)\*\*(?!\*)", r"<b>\1</b>", text, flags=re.DOTALL)

    # Italic: *text*
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text, flags=re.DOTALL
    )

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text, flags=re.DOTALL)

    # Restore placeholders
    for idx, original in enumerate(placeholders):
        ph = f"\x00PH{idx}\x00"
        # Process the original content
        if original.startswith("```"):
            # Fenced code block
            m = re.match(r"```(?:\w+)?\n(.*?)\n```", original, flags=re.DOTALL)
            if m:
                content = m.group(1)
            else:
                m2 = re.match(r"```(.*?)```", original, flags=re.DOTALL)
                content = m2.group(1) if m2 else original
            content = (
                content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            text = text.replace(ph, f"<pre>{content}</pre>")
        else:
            # Inline code
            m = re.match(r"`([^`]+)`", original)
            content = m.group(1) if m else original
            content = (
                content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            text = text.replace(ph, f"<code>{content}</code>")

    return text


def _split_text(text: str, max_length: int = _TELEGRAM_MAX_LENGTH) -> list[str]:
    """Split text into chunks <= ``max_length``, preserving readability.

    Tries to break at paragraph boundaries (``\\n\\n``), then line breaks
    (``\\n``), then spaces, and finally hard-cuts when nothing else works.
    """
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    while len(text) > max_length:
        cut = max_length
        for separator in ["\n\n", "\n", " "]:
            idx = text.rfind(separator, 0, max_length)
            if idx >= max_length // 2:
                cut = idx + len(separator)
                break
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text.strip() if text.strip() else text)
    return parts
