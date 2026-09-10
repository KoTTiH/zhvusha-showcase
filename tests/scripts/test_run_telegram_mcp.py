"""Launcher contract for the personal Telegram MCP integration."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_launch_plan_maps_zhvusha_personal_session_to_upstream_env(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")

    plan = build_launch_plan(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            "TELEGRAM_MCP_SESSION_STRING_PERSONAL": "session-secret",
            "TELEGRAM_MCP_ALLOWED_ROOTS": str(tmp_path / "downloads"),
        }
    )

    assert plan.command == "uv"
    assert plan.args[:4] == ("uv", "--directory", str(checkout), "run")
    assert plan.args[4:] == ("main.py", str(tmp_path / "downloads"))
    assert plan.env["TELEGRAM_SESSION_STRING_PERSONAL"] == "session-secret"
    assert "TELEGRAM_MCP_SESSION_STRING_PERSONAL" not in plan.env
    assert "BOT_TOKEN" not in plan.env


def test_launch_plan_accepts_upstream_personal_session_env(tmp_path: Path) -> None:
    from scripts.run_telegram_mcp import build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")

    plan = build_launch_plan(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            "TELEGRAM_SESSION_STRING_PERSONAL": "session-secret",
        }
    )

    assert plan.env["TELEGRAM_SESSION_STRING_PERSONAL"] == "session-secret"


def test_launch_plan_resolves_uv_from_service_home_when_path_is_minimal(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")
    fake_home = tmp_path / "home"
    fake_uv = fake_home / ".local" / "bin" / "uv"
    fake_uv.parent.mkdir(parents=True)
    fake_uv.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    plan = build_launch_plan(
        {
            "HOME": str(fake_home),
            "PATH": "/usr/bin",
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            "TELEGRAM_SESSION_STRING_PERSONAL": "session-secret",
        }
    )

    assert plan.command == str(fake_uv)
    assert plan.args[:4] == (str(fake_uv), "--directory", str(checkout), "run")


def test_launch_plan_maps_personal_session_name_to_upstream_env(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")
    session_file = tmp_path / "personal.session"
    session_file.write_text("", encoding="utf-8")

    plan = build_launch_plan(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            "TELEGRAM_MCP_SESSION_NAME_PERSONAL": str(session_file),
        }
    )

    assert plan.env["TELEGRAM_SESSION_NAME_PERSONAL"] == str(session_file)
    assert "TELEGRAM_MCP_SESSION_NAME_PERSONAL" not in plan.env


def test_launch_plan_can_reuse_existing_telethon_session_path(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")
    session_file = tmp_path / "zhvusha.session"
    session_file.write_text("", encoding="utf-8")

    plan = build_launch_plan(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            "TELETHON_SESSION_PATH": str(session_file),
        }
    )

    assert plan.env["TELEGRAM_SESSION_NAME_PERSONAL"] == str(session_file)


def test_launch_plan_fails_without_personal_session(tmp_path: Path) -> None:
    from scripts.run_telegram_mcp import TelegramMCPConfigError, build_launch_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "main.py").write_text("print('server')\n", encoding="utf-8")

    with pytest.raises(TelegramMCPConfigError, match="PERSONAL"):
        build_launch_plan(
            {
                "TELEGRAM_API_ID": "12345",
                "TELEGRAM_API_HASH": "hash-value",
                "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
            }
        )


def test_launcher_process_ownership_blocks_second_live_mcp_owner(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import (
        TelegramMCPConfigError,
        acquire_telegram_mcp_process_ownership,
    )
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid in {111, 222},
    )
    first_owner = acquire_telegram_mcp_process_ownership(
        {"WORKSPACE_PATH": str(tmp_path)},
        pid=111,
        owner_id="mcp-a",
        guard=guard,
    )

    with pytest.raises(TelegramMCPConfigError, match="telegram_mcp: already_owned"):
        acquire_telegram_mcp_process_ownership(
            {"WORKSPACE_PATH": str(tmp_path)},
            pid=222,
            owner_id="mcp-b",
            guard=guard,
        )

    owner = guard.status("telegram_mcp").owner
    assert first_owner == "mcp-a"
    assert owner is not None
    assert owner.owner_id == "mcp-a"


def test_launcher_process_ownership_uses_workspace_path(
    tmp_path: Path,
) -> None:
    from scripts.run_telegram_mcp import acquire_telegram_mcp_process_ownership

    workspace = tmp_path / "workspace"

    owner_id = acquire_telegram_mcp_process_ownership(
        {"WORKSPACE_PATH": str(workspace)},
        pid=111,
        owner_id="mcp-a",
    )

    lease_path = workspace / "runtime" / "process-owners.json"
    assert owner_id == "mcp-a"
    assert lease_path.exists()
    assert "telegram_mcp" in lease_path.read_text(encoding="utf-8")


def test_session_generator_uses_same_checkout_and_git_source(tmp_path: Path) -> None:
    from scripts.generate_telegram_mcp_session import build_session_generator_plan

    checkout = tmp_path / "telegram-mcp"
    checkout.mkdir()
    (checkout / "session_string_generator.py").write_text(
        "print('session')\n",
        encoding="utf-8",
    )

    plan = build_session_generator_plan(
        {
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "hash-value",
            "TELEGRAM_MCP_CHECKOUT_PATH": str(checkout),
        }
    )

    assert plan.args == (
        "uv",
        "--directory",
        str(checkout),
        "run",
        "session_string_generator.py",
    )
