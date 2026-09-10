"""Unit tests for src.utils.subprocess_env."""

import os
from unittest.mock import patch

from src.utils.subprocess_env import (
    clean_env_for_claude_cli,
    clean_env_for_codex_cli,
    clean_env_for_git_subprocess,
)


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "ANTHROPIC_API_KEY": "fake_anthropic_secret",
    },
    clear=True,
)
def test_clean_env_removes_anthropic_api_key():
    env = clean_env_for_claude_cli()
    assert "ANTHROPIC_API_KEY" not in env
    assert "BOT_TOKEN" in env


@patch.dict(
    os.environ,
    {"BOT_TOKEN": "fake"},
    clear=True,
)
def test_clean_env_noop_when_key_absent():
    env = clean_env_for_claude_cli()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["BOT_TOKEN"] == "fake"


@patch.dict(
    os.environ,
    {"BOT_TOKEN": "fake", "ANTHROPIC_API_KEY": "fake_anthropic_secret"},
    clear=True,
)
def test_clean_env_does_not_mutate_os_environ():
    env = clean_env_for_claude_cli()
    assert "ANTHROPIC_API_KEY" not in env
    # The mutation happens on a copy; os.environ itself is untouched.
    assert os.environ.get("ANTHROPIC_API_KEY") == "fake_anthropic_secret"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "OPENAI_API_KEY": "fake_openai_secret",
        "OPENAI_BASE_URL": "https://api.openai.example",
        "OPENAI_ORG_ID": "org-test",
        "OPENAI_PROJECT": "proj-test",
    },
    clear=True,
)
def test_clean_env_for_codex_cli_removes_openai_api_vars():
    env = clean_env_for_codex_cli()
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env
    assert "OPENAI_ORG_ID" not in env
    assert "OPENAI_PROJECT" not in env
    assert env["BOT_TOKEN"] == "fake"
    assert os.environ.get("OPENAI_API_KEY") == "fake_openai_secret"


@patch.dict(
    os.environ,
    {
        "BOT_TOKEN": "fake",
        "GIT_INDEX_FILE": "/outer/.git/index",
        "GIT_DIR": "/outer/.git",
        "GIT_WORK_TREE": "/outer",
        "GIT_PREFIX": "src/",
    },
    clear=True,
)
def test_clean_env_for_git_subprocess_removes_repo_scoped_git_vars():
    env = clean_env_for_git_subprocess()
    assert env["BOT_TOKEN"] == "fake"
    assert "GIT_INDEX_FILE" not in env
    assert "GIT_DIR" not in env
    assert "GIT_WORK_TREE" not in env
    assert "GIT_PREFIX" not in env
    assert os.environ["GIT_INDEX_FILE"] == "/outer/.git/index"
