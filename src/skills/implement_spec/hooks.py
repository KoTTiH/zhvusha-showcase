"""PreToolUse hook factories for ImplementSpecSkill (Phase 13).

Two pure factories return ``(tool_name, tool_input) -> HookDecision``
callables. The decisions are framework-neutral; the Editor backend can reuse
the same whitelist and shell safe-list rules. Keeping the logic pure means it
can be unit-tested without booting the backend.

Trust model:

* ``make_edit_write_hook`` — only paths in ``whitelist_paths`` (the
  spec's surgical change surface) may be touched. Everything else
  bounces with a "not in whitelist" reason. Read-only tools (``Read``,
  ``Grep``, ``Glob``) are out of scope and pass through unchanged.
* ``make_bash_hook`` — a tight safe-list: test/lint/typecheck/import
  validators, read-only ``git`` introspection, and ``python -c "..."``
  smoke-imports. Any shell metacharacter (``;&|`$<>``) is rejected so
  the agent can't compose its way out of the safe-list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})
_NOTEBOOK_TOOLS = frozenset({"NotebookEdit"})

_FORBIDDEN_BASH_METACHARS = ";&|`$<>\n"

_SAFE_BASH_RE = re.compile(
    r"""
    ^                                                # start of command
    (?:
        (?:uv\ run\ )?(?:pytest|ruff|mypy|lint-imports)(?:\ .*)?
        |
        git\ (?:status|diff|log|show)(?:\ .*)?
        |
        python\ -c\ "[^"]*"
    )
    $                                                # end of command
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class HookDecision:
    """Hook verdict carried back to the SDK adapter.

    ``allowed=True`` ⇒ proceed; ``allowed=False`` carries a Russian-
    language ``reason`` for chat-side surfacing and audit logs.
    """

    allowed: bool
    reason: str | None = None


def make_edit_write_hook(
    *,
    whitelist_paths: list[str],
    project_root: Path,
    existing_tests_to_update_paths: list[str] | None = None,
) -> Callable[[str, dict[str, Any]], HookDecision]:
    """Build a path-whitelist hook for Edit/Write/MultiEdit/NotebookEdit.

    Absolute ``file_path`` values are normalised to repo-relative before
    the comparison; paths outside ``project_root`` are rejected outright.

    ``existing_tests_to_update_paths`` (Phase 16) extends the allowed
    set with paths the spec has explicitly declared as legitimate
    test mutations (see ``SpecModel.existing_tests_to_update``). The
    hook is path-level only — the prompt-level ``allowed_changes``
    envelope is enforced softly by the system prompt; the hook just
    has to stop denying the path.
    """
    allowed_set = frozenset(
        list(whitelist_paths) + list(existing_tests_to_update_paths or [])
    )
    root = project_root.resolve()

    def hook(tool_name: str, tool_input: dict[str, Any]) -> HookDecision:
        if tool_name not in _EDIT_TOOLS and tool_name not in _NOTEBOOK_TOOLS:
            return HookDecision(allowed=True)
        key = "notebook_path" if tool_name in _NOTEBOOK_TOOLS else "file_path"
        raw = tool_input.get(key)
        if not raw or not isinstance(raw, str):
            return HookDecision(
                allowed=False,
                reason=f"{tool_name}: пустой {key}",
            )
        from pathlib import Path as _Path

        path = _Path(raw)
        if path.is_absolute():
            try:
                rel = str(path.resolve().relative_to(root))
            except ValueError:
                return HookDecision(
                    allowed=False,
                    reason=f"{tool_name}: путь {raw} вне project_root",
                )
        else:
            rel = str(path)
        if rel in allowed_set:
            return HookDecision(allowed=True)
        return HookDecision(
            allowed=False,
            reason=f"{tool_name}: путь {rel} не в whitelist_paths",
        )

    return hook


def make_bash_hook() -> Callable[[str, dict[str, Any]], HookDecision]:
    """Build a Bash safe-list hook (test/lint/typecheck/git-readonly/python-c).

    The metacharacter ban ``;&|`$<>`` is the load-bearing safety
    guarantee — without it an agent could chain ``pytest; rm -rf src``
    past any safe-list.
    """

    def hook(tool_name: str, tool_input: dict[str, Any]) -> HookDecision:
        if tool_name != "Bash":
            return HookDecision(allowed=True)
        cmd = tool_input.get("command", "")
        if not isinstance(cmd, str) or not cmd.strip():
            return HookDecision(allowed=False, reason="Bash: пустая команда")
        if any(c in cmd for c in _FORBIDDEN_BASH_METACHARS):
            return HookDecision(
                allowed=False,
                reason=f"Bash: запрещённые shell metachars в '{cmd[:80]}'",
            )
        if not _SAFE_BASH_RE.match(cmd):
            return HookDecision(
                allowed=False,
                reason=f"Bash: команда '{cmd[:80]}' не в safe-list",
            )
        return HookDecision(allowed=True)

    return hook
