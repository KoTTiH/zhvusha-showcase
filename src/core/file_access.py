"""Secure file reading service for LLM-planned file access."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from src.utils.file_index import list_files

logger = structlog.get_logger()

# Shared workspace security constants
WS_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
WS_TEXT_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv"})

MAX_WORKSPACE_FILES = 20
MAX_CODE_FILES = 3
MAX_FILE_CHARS = 10_000


@dataclass
class FileReadResult:
    """Result of reading requested files."""

    workspace_contents: dict[str, str] = field(default_factory=dict)
    code_contents: dict[str, str] = field(default_factory=dict)


class FileAccessService:
    """Reads workspace and project files with path validation and limits."""

    def __init__(self, workspace_root: Path, project_root: Path) -> None:
        self._workspace = workspace_root
        self._project = project_root

    def get_workspace_index(self) -> str:
        """Return cached file listing of the workspace."""
        return list_files(self._workspace)

    def get_project_index(self) -> str:
        """Return cached file listing of the project."""
        return list_files(self._project)

    def read_files(
        self,
        workspace_files: list[str] | None = None,
        code_files: list[str] | None = None,
    ) -> FileReadResult:
        """Read requested files with security validation and limits."""
        result = FileReadResult()

        for rel in (workspace_files or [])[:MAX_WORKSPACE_FILES]:
            content = self._safe_read(rel, self._workspace)
            if content is not None:
                result.workspace_contents[rel] = content

        for rel in (code_files or [])[:MAX_CODE_FILES]:
            content = self._safe_read(rel, self._project)
            if content is not None:
                result.code_contents[rel] = content

        return result

    def _safe_read(self, relative: str, root: Path) -> str | None:
        """Read a file safely within root boundaries."""
        if ".." in relative:
            logger.warning("path_traversal_blocked", path=relative)
            return None

        raw = root / relative
        if raw.is_symlink():
            logger.warning("symlink_blocked", path=relative)
            return None

        full = raw.resolve()
        root_prefix = str(root.resolve()) + os.sep
        if full != root.resolve() and not str(full).startswith(root_prefix):
            logger.warning("path_escape_blocked", path=relative)
            return None

        try:
            content = full.read_text(encoding="utf-8")
            return content[:MAX_FILE_CHARS]
        except OSError:
            return None


def safe_path(base: Path, requested: str) -> Path | None:
    """Resolve requested path under base, preventing directory traversal and symlinks."""
    raw = base / requested
    if raw.is_symlink():
        return None
    resolved = raw.resolve()
    base_resolved = str(base.resolve()) + os.sep
    if resolved != base.resolve() and not str(resolved).startswith(base_resolved):
        return None
    return resolved


def grep_text_files(  # noqa: C901
    search_root: Path, ws_root: Path, query_lower: str, limit: int
) -> list[dict[str, Any]]:
    """Grep text files under search_root for query. Returns match dicts.

    Skips symlinks, hidden files, non-text files, and files over WS_MAX_FILE_SIZE.
    """
    matches: list[dict[str, Any]] = []
    for file in search_root.rglob("*"):
        if len(matches) >= limit:
            break
        if not file.is_file() or file.is_symlink():
            continue
        try:
            rel_to_search = file.relative_to(search_root)
            rel_to_ws = file.relative_to(ws_root)
        except ValueError:
            continue
        if any(p.startswith(".") for p in rel_to_search.parts):
            continue
        try:
            too_large = file.stat().st_size > WS_MAX_FILE_SIZE
        except OSError:
            continue
        if file.suffix not in WS_TEXT_SUFFIXES or too_large:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if query_lower in line.lower():
                matches.append(
                    {
                        "file": str(rel_to_ws),
                        "line": i,
                        "text": line.strip()[:200],
                    }
                )
                if len(matches) >= limit:
                    break
    return matches


def scan_directory(
    target: Path, relative_root: Path | None = None
) -> list[dict[str, Any]]:
    """Scan directory entries (flat, non-recursive). Skips hidden files and symlinks.

    Raises OSError if target cannot be listed (e.g. PermissionError).
    """
    items: list[dict[str, Any]] = []
    for child in sorted(target.iterdir()):
        if child.name.startswith(".") or child.is_symlink():
            continue
        entry: dict[str, Any] = {
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
        }
        if relative_root is not None:
            entry["path"] = str(child.relative_to(relative_root))
        if child.is_file():
            try:
                entry["size"] = child.stat().st_size
            except OSError:
                continue
        items.append(entry)
    return items
