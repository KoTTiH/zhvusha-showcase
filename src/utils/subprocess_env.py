"""Subprocess environment helpers — pure stdlib, no ``src/*`` imports.

Leaf module. Kept under ``src.utils`` so CLI adapters and delegated sessions can
share environment hardening without reaching into each other's private modules.
"""

from __future__ import annotations

import os

_CODEX_API_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "CODEX_API_KEY",
)

_GIT_REPO_ENV_VARS: tuple[str, ...] = (
    "GIT_INDEX_FILE",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def clean_env_for_claude_cli() -> dict[str, str]:
    """Return a copy of ``os.environ`` with ``ANTHROPIC_API_KEY`` removed.

    Claude CLI authenticates via OAuth when ``ANTHROPIC_API_KEY`` is absent;
    if the key is present, the CLI prefers it, which defeats the purpose of
    using the CLI for strategist-tier subscription pricing.

    Used only by ``src.llm.claude_cli`` as an explicit legacy LLM adapter.
    Self-coding and morning workspace automation use Codex subprocess helpers
    instead.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def clean_env_for_codex_cli() -> dict[str, str]:
    """Return a copy of ``os.environ`` without OpenAI API-routing variables.

    Codex CLI should use the user's Codex/ChatGPT subscription login for this
    project, not an accidentally inherited API key from the process manager or
    shell. Auth state in ``CODEX_HOME`` is preserved; only API-key style
    routing variables are removed.
    """
    env = os.environ.copy()
    for key in _CODEX_API_ENV_VARS:
        env.pop(key, None)
    return env


def clean_env_for_git_subprocess() -> dict[str, str]:
    """Return an env safe for nested git calls in temporary repositories.

    Git hooks and pre-commit can export repository-scoped variables such as
    ``GIT_INDEX_FILE``. If a self-coding test or worker then spawns ``git`` in a
    different tmp repo, those inherited variables point at the outer repo's
    index and can corrupt or break worktree operations.
    """
    env = os.environ.copy()
    for key in _GIT_REPO_ENV_VARS:
        env.pop(key, None)
    return env
