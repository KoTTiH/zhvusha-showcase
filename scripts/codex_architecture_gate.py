#!/usr/bin/env python3
"""Codex Stop hook for ZHVUSHA architecture-boundary verification."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARCHITECTURE_SOURCE_PREFIXES = (
    "src/bot/",
    "src/skills/",
    "src/agent_runtime/",
)
ARCHITECTURE_DOC_PATHS = {
    "AGENTS.md",
    "CLAUDE.md",
    "docs/agent-runtime-principles.md",
    "docs/architecture-invariants.md",
}
BEHAVIOR_CHECK_RE = re.compile(r"(^|\s)(uv\s+run\s+)?(python\s+-m\s+)?pytest(\s|$)")
STATIC_CHECK_RE = re.compile(r"(^|\s)(uv\s+run\s+)?(ruff|mypy)(\s|$)")
PATCH_FILE_RE = re.compile(
    r"^\*\*\* (?:Update|Add|Delete) File: (?P<path>.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class GateDecision:
    """Architecture gate decision for one Codex turn."""

    should_continue: bool
    reason: str = ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    cwd = Path(str(payload.get("cwd") or ".")).resolve()
    turn_id = str(payload.get("turn_id") or "")
    transcript_path = _optional_path(payload.get("transcript_path"))
    touched_paths = touched_paths_for_turn(transcript_path, turn_id, cwd)
    commands = commands_for_turn(transcript_path, turn_id)
    decision = evaluate_gate(touched_paths=touched_paths, commands=commands)
    if decision.should_continue:
        return 0

    print(json.dumps({"decision": "block", "reason": decision.reason}))
    return 0


def evaluate_gate(
    *,
    touched_paths: set[str],
    commands: list[str],
) -> GateDecision:
    """Return whether Codex can stop after this turn."""

    source_changes = {
        path
        for path in touched_paths
        if _is_architecture_source(path) and not path.endswith("/AGENTS.md")
    }
    if not source_changes:
        return GateDecision(should_continue=True)

    has_behavior_check = any(BEHAVIOR_CHECK_RE.search(command) for command in commands)
    has_static_check = any(STATIC_CHECK_RE.search(command) for command in commands)
    if has_behavior_check:
        return GateDecision(should_continue=True)

    changed = ", ".join(sorted(source_changes))
    if has_static_check:
        return GateDecision(
            should_continue=False,
            reason=(
                "Architecture gate: changed ZHVUSHA body/orchestration/runtime "
                f"source ({changed}) but no pytest contract/regression check ran "
                "in this turn. Run a focused pytest for the affected boundary or "
                "explain and add the missing contract before final."
            ),
        )
    return GateDecision(
        should_continue=False,
        reason=(
            "Architecture gate: changed ZHVUSHA body/orchestration/runtime "
            f"source ({changed}) without a behavior verification command. "
            "Identify the affected boundary, avoid scenario-specific fixes, and "
            "run focused pytest before final."
        ),
    )


def touched_paths_for_turn(
    transcript_path: Path | None,
    turn_id: str,
    cwd: Path,
) -> set[str]:
    """Collect files changed by apply_patch events in the current turn."""

    touched: set[str] = set()
    if transcript_path is not None and transcript_path.exists():
        for item in _iter_jsonl(transcript_path):
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if turn_id and _payload_turn_id(payload) not in {"", turn_id}:
                continue
            touched.update(_paths_from_payload(payload, cwd))
    return touched


def commands_for_turn(transcript_path: Path | None, turn_id: str) -> list[str]:
    """Collect shell commands issued in the current turn."""

    commands: list[str] = []
    if transcript_path is None or not transcript_path.exists():
        return commands
    for item in _iter_jsonl(transcript_path):
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        if turn_id and _payload_turn_id(payload) not in {"", turn_id}:
            continue
        name = payload.get("name")
        if name not in {"exec_command", "shell"}:
            continue
        arguments = _json_object(payload.get("arguments"))
        command = arguments.get("cmd") or arguments.get("command")
        if isinstance(command, str) and command.strip():
            commands.append(command)
    return commands


def _paths_from_payload(payload: dict[str, Any], cwd: Path) -> set[str]:
    paths: set[str] = set()
    changes = payload.get("changes")
    if isinstance(changes, dict):
        for raw_path in changes:
            paths.add(_normalize_path(raw_path, cwd))
    if payload.get("name") == "apply_patch":
        patch_input = payload.get("input")
        if isinstance(patch_input, str):
            paths.update(
                _normalize_path(match.group("path"), cwd)
                for match in PATCH_FILE_RE.finditer(patch_input)
            )
    return {path for path in paths if path}


def _payload_turn_id(payload: dict[str, Any]) -> str:
    raw = payload.get("turn_id")
    return raw if isinstance(raw, str) else ""


def _normalize_path(raw_path: str, cwd: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(cwd).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _is_architecture_source(path: str) -> bool:
    if path in ARCHITECTURE_DOC_PATHS:
        return False
    return path.startswith(ARCHITECTURE_SOURCE_PREFIXES) and path.endswith(".py")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return items
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def _json_object(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_path(raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    return Path(raw)


if __name__ == "__main__":
    raise SystemExit(main())
