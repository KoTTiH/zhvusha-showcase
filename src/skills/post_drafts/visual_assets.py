"""Frontmatter helpers for approved channel visual assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def approve_visual_asset(
    visual: dict[str, Any],
    *,
    workspace_root: Path,
    asset_path: str,
    caption: str = "",
) -> dict[str, Any]:
    """Mark an already prepared workspace artifact as approved for publish."""
    resolved = resolve_workspace_asset(
        workspace_root=workspace_root, asset_path=asset_path
    )
    updated = dict(visual)
    updated["status"] = "approved"
    updated["asset_path"] = resolved.relative_to(
        workspace_root.expanduser().resolve()
    ).as_posix()
    if caption:
        updated["caption"] = caption[:1024]
    return updated


def resolve_workspace_asset(*, workspace_root: Path, asset_path: str) -> Path:
    """Resolve a relative artifact path and require workspace containment."""
    if not asset_path.strip():
        raise ValueError("asset_path is required")
    raw = Path(asset_path)
    if raw.is_absolute():
        raise ValueError("asset_path must be relative to workspace_root")
    root = workspace_root.expanduser().resolve()
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        raise ValueError("asset_path escapes workspace_root")
    if not target.is_file():
        raise FileNotFoundError(asset_path)
    return target
