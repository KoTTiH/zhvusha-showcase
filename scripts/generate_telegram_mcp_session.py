#!/usr/bin/env python3
"""Run upstream telegram-mcp session_string_generator.py from the pinned checkout."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from scripts.run_telegram_mcp import (
    LaunchPlan,
    TelegramMCPConfigError,
    _build_base_child_env,
    _checkout_path,
    _env_value,
    _require_file,
    _require_telegram_api_credentials,
    load_runtime_env,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_session_generator_plan(
    source_env: Mapping[str, str] | None = None,
) -> LaunchPlan:
    """Build the command/env for interactive personal session generation."""
    env = dict(load_runtime_env() if source_env is None else source_env)
    uv_path = _env_value(env, "TELEGRAM_MCP_UV_PATH", "uv")
    checkout = _checkout_path(env)
    _require_file(
        checkout / "session_string_generator.py",
        "telegram-mcp checkout must contain session_string_generator.py",
    )
    api_id, api_hash = _require_telegram_api_credentials(env)

    child_env = _build_base_child_env(env)
    child_env["TELEGRAM_API_ID"] = api_id
    child_env["TELEGRAM_API_HASH"] = api_hash
    args = (
        uv_path,
        "--directory",
        str(checkout),
        "run",
        "session_string_generator.py",
    )
    return LaunchPlan(command=uv_path, args=args, env=child_env)


def main() -> int:
    """Replace this process with the upstream session generator."""
    try:
        plan = build_session_generator_plan()
    except TelegramMCPConfigError as exc:
        print(f"Telegram MCP config error: {exc}", file=sys.stderr)
        return 2
    os.execvpe(plan.command, plan.args, plan.env)  # noqa: S606
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
