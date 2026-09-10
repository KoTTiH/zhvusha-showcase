"""Registry for the self-coding code-agent backend.

Self-coding is Codex-only. Legacy Claude automation is intentionally not a
fallback because it can trip account-safety limits when used as a coding CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    ArchitectRequest,
    CodeAgentBackend,
    CodeAgentResult,
    CodeAgentUnavailableError,
    EditorRequest,
    ExplorerRequest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = structlog.get_logger()

CODEX_BACKEND = "codex_cli"
_BLOCKED_BACKENDS = frozenset({"claude_agent_sdk", "claude_code_sdk", "claude_cli"})


class CodeAgentRegistry:
    """Resolve the single allowed self-coding backend."""

    def __init__(
        self,
        *,
        backends: Mapping[str, CodeAgentBackend],
        backend: str,
    ) -> None:
        if backend in _BLOCKED_BACKENDS:
            raise ValueError(
                "Claude automation is disabled for self-coding; use codex_cli."
            )
        self._backends = dict(backends)
        self._backend = backend

    @property
    def backend_order(self) -> tuple[str, ...]:
        """Compatibility shape: one backend, no fallback chain."""
        return (self._backend,)

    async def run_architect(self, request: ArchitectRequest) -> CodeAgentResult:
        backend = self._get_backend()
        result = await backend.run_architect(request)
        logger.info("code_agent_architect_done", backend=result.backend)
        return result

    async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
        backend = self._get_backend()
        result = await backend.run_explorer(request)
        logger.info("code_agent_explorer_done", backend=result.backend)
        return result

    async def run_editor(self, request: EditorRequest) -> CodeAgentResult:
        backend = self._get_backend()
        result = await backend.run_editor(request)
        logger.info("code_agent_editor_done", backend=result.backend)
        return result

    def _get_backend(self) -> CodeAgentBackend:
        backend = self._backends.get(self._backend)
        if backend is None:
            raise CodeAgentUnavailableError(
                self._backend,
                "configured self-coding backend is not registered",
            )
        return backend


def build_codex_registry(
    *,
    backend: str,
    codex_path: str,
    codex_model: str,
    reasoning_effort: str = "",
    timeout_seconds: float = 7200.0,
) -> CodeAgentRegistry:
    """Build the production Codex-only registry."""
    return CodeAgentRegistry(
        backends={
            CODEX_BACKEND: CodexCliBackend(
                codex_path=codex_path,
                model=codex_model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
        },
        backend=backend,
    )
