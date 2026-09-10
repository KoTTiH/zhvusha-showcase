"""Cached file index for LLM prompts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        ".processed",
    }
)

_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 60.0


def list_files(root: Path, max_depth: int = 3) -> str:
    """Recursive file listing with 60s in-memory cache."""
    key = str(root.resolve())
    now = time.monotonic()
    if key in _cache:
        cached_at, result = _cache[key]
        if now - cached_at < _CACHE_TTL:
            return result

    lines: list[str] = []
    _scan(root, root, 0, max_depth, lines)
    result = "\n".join(lines)
    _cache[key] = (now, result)
    return result


def _scan(
    base: Path,
    current: Path,
    depth: int,
    max_depth: int,
    lines: list[str],
) -> None:
    if depth > max_depth:
        return
    try:
        entries = sorted(current.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
            continue
        rel = entry.relative_to(base)
        indent = "  " * depth
        if entry.is_dir():
            lines.append(f"{indent}{rel}/")
            _scan(base, entry, depth + 1, max_depth, lines)
        elif entry.is_file():
            lines.append(f"{indent}{rel}")


def clear_cache() -> None:
    """Clear the file index cache (for testing)."""
    _cache.clear()
