"""Provider-neutral request/result contracts for self-coding code agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@dataclass(frozen=True)
class CodeAgentResult:
    """Text summary returned by the concrete code-agent runtime."""

    text: str
    backend: str = "codex_cli"
    session_id: str = ""


class CodeAgentUnavailableError(RuntimeError):
    """Raised when the selected code-agent backend cannot be started."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend}: {reason}")


class CodeAgentExecutionError(RuntimeError):
    """Raised when the backend starts but the session itself fails."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend}: {reason}")


@dataclass(frozen=True)
class ArchitectRequest:
    """Read-only request for spec drafting."""

    system_prompt: str
    user_prompt: str
    cwd: Path
    model: str = ""
    reasoning_effort: str = ""
    session_id: str = ""
    persist_session: bool = False


@dataclass(frozen=True)
class ExplorerRequest:
    """Read-only request for code/session investigation without spec creation."""

    system_prompt: str
    user_prompt: str
    cwd: Path
    progress_callback: Callable[[str], Awaitable[None]] | None = None
    model: str = ""
    reasoning_effort: str = ""
    session_id: str = ""
    persist_session: bool = False


@dataclass(frozen=True)
class EditorRequest:
    """Write-capable request for implementing an approved spec."""

    system_prompt: str
    user_prompt: str
    cwd: Path
    project_root: Path
    whitelist_paths: list[str]
    existing_tests_to_update_paths: list[str] = field(default_factory=list)
    progress_callback: Callable[[str], Awaitable[None]] | None = None
    model: str = ""
    reasoning_effort: str = ""
    session_id: str = ""
    persist_session: bool = False


@dataclass(frozen=True)
class DelegateRequest:
    """Free-form delegated code-agent request."""

    task: str
    cwd: Path
    model: str = ""
    reasoning_effort: str = ""
    session_id: str = ""
    persist_session: bool = False


class CodeAgentBackend(Protocol):
    """Backend interface consumed by self-coding skills."""

    name: str

    async def run_architect(self, request: ArchitectRequest) -> CodeAgentResult:
        """Draft a spec from a shared Architect request."""
        ...

    async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
        """Investigate code/files in read-only mode for discussion."""
        ...

    async def run_editor(self, request: EditorRequest) -> CodeAgentResult:
        """Implement an approved spec from a shared Editor request."""
        ...

    async def run_delegate(self, request: DelegateRequest) -> CodeAgentResult:
        """Run a free-form delegated code-agent task."""
        ...
