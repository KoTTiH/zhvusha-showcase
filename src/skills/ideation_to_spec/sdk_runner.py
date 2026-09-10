"""Codex CLI adapter for the Architect half of self-coding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    ArchitectRequest,
    CodeAgentUnavailableError,
)

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_MODEL = ""

SDKUnavailableError = CodeAgentUnavailableError


async def run_architect_sdk(
    *,
    user_prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str = _DEFAULT_MODEL,
    codex_path: str = "codex",
) -> str:
    """Run Codex in read-only Architect mode and return assistant text."""
    backend = CodexCliBackend(codex_path=codex_path, model=model)
    result = await backend.run_architect(
        ArchitectRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=cwd,
            model=model,
        )
    )
    return result.text
