"""Codex CLI adapter for the Editor half of self-coding.

The public function name stays ``run_editor_sdk`` for compatibility with
existing skill wiring and tests, but the implementation no longer starts any
legacy coding automation. All shared Editor rules are carried in
``EditorRequest`` and executed by the Codex backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    CodeAgentResult,
    CodeAgentUnavailableError,
    EditorRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

_DEFAULT_MODEL = ""

EditorSdkResult = CodeAgentResult
SDKUnavailableError = CodeAgentUnavailableError


async def run_editor_sdk(
    *,
    user_prompt: str,
    system_prompt: str,
    cwd: Path,
    project_root: Path,
    whitelist_paths: list[str],
    existing_tests_to_update_paths: list[str] | None = None,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
    model: str = _DEFAULT_MODEL,
    codex_path: str = "codex",
    session_id: str = "",
    persist_session: bool = False,
) -> EditorSdkResult:
    """Run a Codex Editor session bounded by the spec whitelist.

    The final commit gate still enforces the whitelist after the backend
    returns; the prompt carries the same whitelist rules to the agent before it
    writes.
    """
    backend = CodexCliBackend(codex_path=codex_path, model=model)
    return await backend.run_editor(
        EditorRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=cwd,
            project_root=project_root,
            whitelist_paths=whitelist_paths,
            existing_tests_to_update_paths=list(existing_tests_to_update_paths or []),
            progress_callback=progress_callback,
            model=model,
            session_id=session_id,
            persist_session=persist_session,
        )
    )
