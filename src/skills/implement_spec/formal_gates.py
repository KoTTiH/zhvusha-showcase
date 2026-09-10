"""Formal non-LLM gates for self-coding cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_BLOCKED_BACKENDS = frozenset({"claude_agent_sdk", "claude_code_sdk", "claude_cli"})


@dataclass(frozen=True)
class FormalGateInputs:
    baseline_test_count: int
    current_test_count: int
    changed_files: list[str]
    whitelist_paths: list[str]
    existing_tests_to_update_paths: list[str]
    audit_log: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormalGateVerdict:
    passed: bool
    issues: list[str] = field(default_factory=list)


def run_formal_gates(inputs: FormalGateInputs) -> FormalGateVerdict:
    """Run deterministic checks before model-based review."""
    issues: list[str] = []
    if inputs.current_test_count < inputs.baseline_test_count:
        issues.append(
            f"test count decreased: {inputs.baseline_test_count} -> {inputs.current_test_count}"
        )
    allowed = set(inputs.whitelist_paths) | set(inputs.existing_tests_to_update_paths)
    outside = [path for path in inputs.changed_files if path not in allowed]
    if outside:
        issues.append("changed files outside whitelist: " + ", ".join(outside))
    backend = str(inputs.audit_log.get("backend", ""))
    if backend in _BLOCKED_BACKENDS:
        issues.append(f"forbidden Claude backend in audit metadata: {backend}")
    if not inputs.audit_log.get("slug"):
        issues.append("audit log missing slug")
    return FormalGateVerdict(passed=not issues, issues=issues)
