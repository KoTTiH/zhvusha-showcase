from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.codex_limits import load_latest_codex_limit_snapshot


def _write_session(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_load_latest_codex_limit_snapshot_from_session_logs(tmp_path: Path) -> None:
    older = tmp_path / "sessions" / "2026" / "05" / "07" / "older.jsonl"
    newer = tmp_path / "sessions" / "2026" / "05" / "08" / "newer.jsonl"
    _write_session(
        older,
        {
            "type": "event_msg",
            "payload": {
                "rate_limits": {
                    "primary": {"used_percent": 99.0, "window_minutes": 300}
                }
            },
        },
    )
    _write_session(
        newer,
        {
            "type": "event_msg",
            "payload": {
                "rate_limits": {
                    "plan_type": "prolite",
                    "primary": {
                        "used_percent": 6.0,
                        "window_minutes": 300,
                        "resets_at": 1778233294,
                    },
                    "secondary": {
                        "used_percent": 36.0,
                        "window_minutes": 10080,
                        "resets_at": 1778553681,
                    },
                }
            },
        },
    )

    snapshot = load_latest_codex_limit_snapshot(tmp_path)

    assert snapshot is not None
    assert snapshot.primary_used_percent == 6.0
    assert snapshot.primary_window_minutes == 300
    assert snapshot.primary_resets_at == 1778233294
    assert snapshot.secondary_used_percent == 36.0
    assert snapshot.secondary_window_minutes == 10080
    assert snapshot.secondary_resets_at == 1778553681
    assert snapshot.plan_type == "prolite"


def test_load_latest_codex_limit_snapshot_returns_none_without_rate_limits(
    tmp_path: Path,
) -> None:
    _write_session(
        tmp_path / "sessions" / "2026" / "05" / "08" / "empty.jsonl",
        {"type": "event_msg", "payload": {"token_count": {}}},
    )

    assert load_latest_codex_limit_snapshot(tmp_path) is None
