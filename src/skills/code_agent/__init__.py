"""Codex-backed code-agent runtime for delegated self-coding skills."""

from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    ArchitectRequest,
    CodeAgentExecutionError,
    CodeAgentResult,
    CodeAgentUnavailableError,
    DelegateRequest,
    EditorRequest,
)
from src.skills.code_agent.registry import CodeAgentRegistry, build_codex_registry

__all__ = [
    "ArchitectRequest",
    "CodeAgentExecutionError",
    "CodeAgentRegistry",
    "CodeAgentResult",
    "CodeAgentUnavailableError",
    "CodexCliBackend",
    "DelegateRequest",
    "EditorRequest",
    "build_codex_registry",
]
