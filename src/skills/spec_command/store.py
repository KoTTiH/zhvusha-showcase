"""Filesystem store for ``tasks/<YYYY-MM-DD>-<slug>.yaml`` spec files.

Pure I/O — no LLM, no Telegram. Loaded / saved via PyYAML with the order
of top-level keys preserved (``sort_keys=False``) so that human-edited
specs stay diff-friendly across status transitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from src.skills.spec_command.parser import SpecModel

if TYPE_CHECKING:
    from pathlib import Path


def list_spec_files(tasks_dir: Path) -> list[Path]:
    """Return all ``tasks/*.yaml`` paths in modification-time order (newest last)."""
    if not tasks_dir.exists():
        return []
    files = [p for p in tasks_dir.iterdir() if p.suffix == ".yaml" and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def find_spec_path(tasks_dir: Path, slug: str) -> Path | None:
    """Find the spec file matching ``<slug>`` regardless of date prefix."""
    if not tasks_dir.exists():
        return None
    matches = [
        p
        for p in tasks_dir.iterdir()
        if p.suffix == ".yaml" and p.is_file() and p.stem.endswith(f"-{slug}")
    ]
    if not matches:
        # fall back to exact match without date prefix
        candidate = tasks_dir / f"{slug}.yaml"
        return candidate if candidate.is_file() else None
    return matches[0]


def load_spec(path: Path) -> SpecModel:
    """Read and validate a single spec file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SpecModel.model_validate(raw)


def load_spec_raw(path: Path) -> dict[str, Any]:
    """Read a spec file as a plain dict (preserves unknown / extra fields)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"spec file {path} is not a YAML mapping")
    return raw


def save_spec_raw(path: Path, data: dict[str, Any]) -> None:
    """Write a spec dict back, preserving key order (``sort_keys=False``)."""
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
