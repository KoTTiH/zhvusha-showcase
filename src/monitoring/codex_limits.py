"""Read Codex account limits from local Codex session telemetry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_DEFAULT_TAIL_BYTES = 512 * 1024
_DEFAULT_MAX_FILES = 5


@dataclass(frozen=True)
class CodexLimitSnapshot:
    """Last known Codex rate-limit state from Codex session JSONL logs."""

    primary_used_percent: float | None = None
    primary_window_minutes: int | None = None
    primary_resets_at: int | None = None
    secondary_used_percent: float | None = None
    secondary_window_minutes: int | None = None
    secondary_resets_at: int | None = None
    plan_type: str | None = None


def load_latest_codex_limit_snapshot(
    codex_home: Path | None = None,
    *,
    max_files: int = _DEFAULT_MAX_FILES,
    tail_bytes: int = _DEFAULT_TAIL_BYTES,
) -> CodexLimitSnapshot | None:
    """Return the newest Codex limits snapshot available on disk.

    Codex writes runtime ``rate_limits`` into its local session JSONL files.
    There is no public local CLI command for these numbers, so the bot reads
    the same telemetry the TUI status line uses and shows the latest known
    value in its pinned dashboard.
    """
    home = codex_home or Path.home() / ".codex"
    sessions_dir = home / "sessions"
    if not sessions_dir.exists():
        return None

    for path in _latest_session_files(sessions_dir, max_files=max_files):
        snapshot = _load_snapshot_from_session_file(path, tail_bytes=tail_bytes)
        if snapshot is not None:
            return snapshot
    return None


def format_codex_limit_snapshot(snapshot: CodexLimitSnapshot) -> str | None:
    """Render a compact dashboard fragment."""
    parts: list[str] = []
    secondary = _format_window(
        used_percent=snapshot.secondary_used_percent,
        window_minutes=snapshot.secondary_window_minutes,
    )
    if secondary:
        parts.append(secondary)

    primary = _format_window(
        used_percent=snapshot.primary_used_percent,
        window_minutes=snapshot.primary_window_minutes,
    )
    if primary:
        parts.append(primary)

    return " · ".join(parts) if parts else None


def latest_codex_limits_summary() -> str | None:
    snapshot = load_latest_codex_limit_snapshot()
    if snapshot is None:
        return None
    return format_codex_limit_snapshot(snapshot)


def _latest_session_files(sessions_dir: Path, *, max_files: int) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for path in sessions_dir.rglob("*.jsonl"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return [path for _, path in candidates[: max(1, max_files)]]


def _load_snapshot_from_session_file(
    path: Path, *, tail_bytes: int
) -> CodexLimitSnapshot | None:
    try:
        text = _read_tail(path, max_bytes=tail_bytes)
    except OSError:
        logger.debug("codex_limits_session_read_failed", path=str(path), exc_info=True)
        return None

    for line in reversed(text.splitlines()):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        rate_limits = raw.get("payload", {}).get("rate_limits")
        if isinstance(rate_limits, dict):
            snapshot = _snapshot_from_rate_limits(rate_limits)
            if snapshot is not None:
                return snapshot
    return None


def _read_tail(path: Path, *, max_bytes: int) -> str:
    size = path.stat().st_size
    start = max(0, size - max(1, max_bytes))
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    if start > 0:
        _discard, _sep, text = text.partition("\n")
    return text


def _snapshot_from_rate_limits(data: dict[str, Any]) -> CodexLimitSnapshot | None:
    primary = data.get("primary")
    secondary = data.get("secondary")
    if not isinstance(primary, dict) and not isinstance(secondary, dict):
        return None

    plan_type = data.get("plan_type")
    return CodexLimitSnapshot(
        primary_used_percent=_coerce_float(
            primary.get("used_percent") if isinstance(primary, dict) else None
        ),
        primary_window_minutes=_coerce_int(
            primary.get("window_minutes") if isinstance(primary, dict) else None
        ),
        primary_resets_at=_coerce_int(
            primary.get("resets_at") if isinstance(primary, dict) else None
        ),
        secondary_used_percent=_coerce_float(
            secondary.get("used_percent") if isinstance(secondary, dict) else None
        ),
        secondary_window_minutes=_coerce_int(
            secondary.get("window_minutes") if isinstance(secondary, dict) else None
        ),
        secondary_resets_at=_coerce_int(
            secondary.get("resets_at") if isinstance(secondary, dict) else None
        ),
        plan_type=str(plan_type) if plan_type else None,
    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_window(
    *,
    used_percent: float | None,
    window_minutes: int | None,
) -> str | None:
    if used_percent is None or window_minutes is None:
        return None
    label = _format_window_label(window_minutes)
    return f"{label} {_format_percent(used_percent)}"


def _format_window_label(window_minutes: int) -> str:
    if window_minutes == 10080:
        return "7д"
    if window_minutes % 60 == 0:
        return f"{window_minutes // 60}ч"
    return f"{window_minutes}м"


def _format_percent(value: float) -> str:
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"
