#!/usr/bin/env python3
"""Launch the upstream Telegram MCP server for Жвуша's personal account.

The repo-level `.mcp.json` points here so Telegram credentials and session
strings stay in `.env`/host env instead of being committed to MCP config.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import dotenv_values
from src.core.process_guard import FileProcessOwnershipGuard

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_CHECKOUT_PATH = "~/.local/share/zhvusha/telegram-mcp"
DEFAULT_ALLOWED_ROOTS = "~/zhvusha-workspace/telegram-mcp"
DEFAULT_WORKSPACE_PATH = "~/zhvusha-workspace"
DEFAULT_ACCOUNT_LABEL = "personal"
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,30}$")
_PASSTHROUGH_ENV_KEYS = {
    "ALL_PROXY",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "PYTHONPATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "VIRTUAL_ENV",
}
_PASSTHROUGH_ENV_PREFIXES = ("LC_", "TELEGRAM_PROXY_", "UV_")


class TelegramMCPConfigError(RuntimeError):
    """Raised when the Telegram MCP launch configuration is incomplete."""


@dataclass(frozen=True)
class LaunchPlan:
    """Prepared exec plan for the upstream Telegram MCP process."""

    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def load_runtime_env(env_file: Path = Path(".env")) -> dict[str, str]:
    """Load `.env` first and let the real process environment override it."""
    loaded: dict[str, str] = {}
    if env_file.exists():
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                loaded[str(key)] = str(value)
    loaded.update(os.environ)
    return loaded


def build_launch_plan(source_env: Mapping[str, str] | None = None) -> LaunchPlan:
    """Build the command/env used by `.mcp.json` to start telegram-mcp."""
    env = dict(load_runtime_env() if source_env is None else source_env)
    uv_path = _uv_command(env)
    checkout = _checkout_path(env)
    _require_file(
        checkout / "main.py",
        "telegram-mcp checkout must contain main.py",
    )
    api_id, api_hash = _require_telegram_api_credentials(env)
    label = _account_label(env)
    upstream_session_key = f"TELEGRAM_SESSION_STRING_{label}"
    upstream_session_name_key = f"TELEGRAM_SESSION_NAME_{label}"
    session = _env_value(env, f"TELEGRAM_MCP_SESSION_STRING_{label}", "")
    if not session:
        session = _env_value(env, upstream_session_key, "")
    session_name = ""
    if not session:
        session_name = _env_value(env, f"TELEGRAM_MCP_SESSION_NAME_{label}", "")
    if not session and not session_name:
        session_name = _env_value(env, upstream_session_name_key, "")
    if not session and not session_name and label == "PERSONAL":
        session_name = _env_value(env, "TELETHON_SESSION_PATH", "")
    if not session and not session_name:
        raise TelegramMCPConfigError(
            f"missing TELEGRAM_MCP_SESSION_STRING_{label} or "
            f"TELEGRAM_MCP_SESSION_NAME_{label}; generate a personal Telegram "
            "session string first or point to an existing Telethon session file"
        )

    child_env = _build_base_child_env(env)
    child_env["TELEGRAM_API_ID"] = api_id
    child_env["TELEGRAM_API_HASH"] = api_hash
    if session:
        child_env[upstream_session_key] = session
    else:
        child_env[upstream_session_name_key] = str(Path(session_name).expanduser())
    child_env.pop("TELEGRAM_SESSION_STRING", None)
    child_env.pop("TELEGRAM_SESSION_NAME", None)
    for key in tuple(child_env):
        if key.startswith(("TELEGRAM_MCP_SESSION_STRING", "TELEGRAM_MCP_SESSION_NAME")):
            child_env.pop(key)

    args = (
        uv_path,
        "--directory",
        str(checkout),
        "run",
        "main.py",
        *_allowed_roots(env),
    )
    return LaunchPlan(command=uv_path, args=args, env=child_env)


def main() -> int:
    """Replace this process with the upstream stdio MCP server."""
    try:
        acquire_telegram_mcp_process_ownership()
        plan = build_launch_plan()
    except TelegramMCPConfigError as exc:
        print(f"Telegram MCP config error: {exc}", file=sys.stderr)
        return 2
    os.execvpe(plan.command, plan.args, plan.env)  # noqa: S606
    return 1


def acquire_telegram_mcp_process_ownership(
    source_env: Mapping[str, str] | None = None,
    *,
    pid: int | None = None,
    owner_id: str | None = None,
    guard: FileProcessOwnershipGuard | None = None,
) -> str:
    """Acquire the standalone MCP service lease before replacing this process."""
    from src.core.process_guard import render_process_ownership_report

    env = dict(load_runtime_env() if source_env is None else source_env)
    workspace_root = Path(
        _env_value(env, "WORKSPACE_PATH", DEFAULT_WORKSPACE_PATH)
    ).expanduser()
    process_guard = guard or FileProcessOwnershipGuard(
        workspace_root / "runtime" / "process-owners.json"
    )
    resolved_owner = owner_id or _telegram_mcp_owner_id()
    status = process_guard.acquire(
        service="telegram_mcp",
        owner_id=resolved_owner,
        pid=pid,
    )
    if not status.acquired:
        raise TelegramMCPConfigError(render_process_ownership_report((status,)))
    return resolved_owner


def _telegram_mcp_owner_id() -> str:
    return f"telegram-mcp-process:{os.getpid()}:{int(time.time())}"


def _checkout_path(env: Mapping[str, str]) -> Path:
    raw = _env_value(env, "TELEGRAM_MCP_CHECKOUT_PATH", DEFAULT_CHECKOUT_PATH)
    return Path(raw).expanduser().resolve()


def _require_file(path: Path, message: str) -> None:
    if path.is_file():
        return
    raise TelegramMCPConfigError(
        f"{message}: {path}. Clone upstream from "
        "https://github.com/chigwell/telegram-mcp.git"
    )


def _require_telegram_api_credentials(env: Mapping[str, str]) -> tuple[str, str]:
    api_id = _env_value(env, "TELEGRAM_API_ID", "")
    api_hash = _env_value(env, "TELEGRAM_API_HASH", "")
    if not api_id or api_id == "0":
        raise TelegramMCPConfigError("missing TELEGRAM_API_ID from my.telegram.org")
    if not api_hash:
        raise TelegramMCPConfigError("missing TELEGRAM_API_HASH from my.telegram.org")
    return api_id, api_hash


def _account_label(env: Mapping[str, str]) -> str:
    raw = _env_value(env, "TELEGRAM_MCP_ACCOUNT_LABEL", DEFAULT_ACCOUNT_LABEL)
    if not _LABEL_RE.fullmatch(raw):
        raise TelegramMCPConfigError(
            "TELEGRAM_MCP_ACCOUNT_LABEL must be letters, digits or underscores"
        )
    return raw.upper()


def _allowed_roots(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = _env_value(env, "TELEGRAM_MCP_ALLOWED_ROOTS", DEFAULT_ALLOWED_ROOTS)
    if not raw:
        return ()
    separator = os.pathsep if os.pathsep in raw else ","
    roots = []
    for item in raw.split(separator):
        cleaned = item.strip()
        if cleaned:
            roots.append(str(Path(cleaned).expanduser()))
    return tuple(roots)


def _uv_command(env: Mapping[str, str]) -> str:
    raw = _env_value(env, "TELEGRAM_MCP_UV_PATH", "uv")
    if raw != "uv" or "/" in raw:
        return str(Path(raw).expanduser())

    if "PATH" in env:
        resolved = shutil.which(raw, path=env["PATH"])
        if resolved:
            return resolved

    home = env.get("HOME")
    if home:
        local_uv = Path(home).expanduser() / ".local" / "bin" / "uv"
        if local_uv.is_file():
            return str(local_uv)

    return raw


def _build_base_child_env(env: Mapping[str, str]) -> dict[str, str]:
    child: dict[str, str] = {}
    for key, value in env.items():
        if key in _PASSTHROUGH_ENV_KEYS or key.startswith(_PASSTHROUGH_ENV_PREFIXES):
            child[key] = value
    return child


def _env_value(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    return value.strip() if isinstance(value, str) else default


if __name__ == "__main__":
    raise SystemExit(main())
