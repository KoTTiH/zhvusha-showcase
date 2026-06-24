"""Standalone keyword-based domain detection. No external dependencies."""

from __future__ import annotations

import re

_KWORK_KEYWORDS = re.compile(
    r"\b(?:кворк|kwork|фриланс|заказ|бюджет|проект[аеу]?\b.*(?:руб|₽)|клиент)\b",
    re.IGNORECASE,
)
_CONTENT_KEYWORDS = re.compile(
    r"\b(?:пост|канал|контент|блог|статья|публикац)\b",
    re.IGNORECASE,
)
_OUTREACH_KEYWORDS = re.compile(
    r"\b(?:сайт|бот|цена|разработ|лендинг|верстк)\b",
    re.IGNORECASE,
)


def detect_domain(content: str, source: str = "", mode: str = "") -> str:
    """Detect interaction domain from content, source, and mode.

    Priority:
    1. source field (set by a skill): kwork → kwork, channel/morning_session → content
    2. mode: assistant → outreach
    3. Keyword heuristics on content
    4. Default: chat

    Returns: "kwork" | "chat" | "content" | "outreach"
    """
    # Priority 1: explicit source
    if source == "kwork":
        return "kwork"
    if source in ("channel", "morning_session"):
        return "content"

    # Priority 2: mode-based
    if mode == "assistant":
        return "outreach"

    # Priority 3: keyword heuristics
    if _KWORK_KEYWORDS.search(content):
        return "kwork"
    if _CONTENT_KEYWORDS.search(content):
        return "content"
    if _OUTREACH_KEYWORDS.search(content):
        return "outreach"

    # Priority 4: default
    return "chat"
