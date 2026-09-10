"""Mask verbose tool outputs in conversation history to prevent hallucination."""

from __future__ import annotations

import re

_FILE_CONTENT_RE = re.compile(
    r'<FILE_CONTENT\s+source="([^"]+)"[^>]*>.*?</FILE_CONTENT>',
    re.DOTALL,
)

_MEMORY_FACTS_RE = re.compile(
    r"<MEMORY_FACTS>.*?</MEMORY_FACTS>",
    re.DOTALL,
)


def mask_file_contents(text: str) -> str:
    """Replace <FILE_CONTENT> blocks with compact pointers."""
    return _FILE_CONTENT_RE.sub(r"[Прочитан файл: \1]", text)


def mask_memory_facts(text: str) -> str:
    """Replace <MEMORY_FACTS> blocks with compact pointer."""
    return _MEMORY_FACTS_RE.sub("[Были извлечены факты из памяти]", text)


def mask_observations(text: str) -> str:
    """Strip all verbose retrieval artifacts from text."""
    text = mask_file_contents(text)
    return mask_memory_facts(text)
