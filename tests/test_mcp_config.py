"""Repository MCP client configuration contract."""

from __future__ import annotations

import json
from pathlib import Path


def test_mcp_config_registers_telegram_mcp_personal_without_secrets() -> None:
    config = json.loads(Path(".mcp.json").read_text(encoding="utf-8"))

    servers = config["mcpServers"]
    assert "zhvusha-knowledge" in servers
    assert "telegram-mcp-personal" in servers

    telegram = servers["telegram-mcp-personal"]
    assert telegram["command"].endswith("python")
    assert telegram["args"] == ["scripts/run_telegram_mcp.py"]

    raw_config = json.dumps(config)
    assert "TELEGRAM_SESSION_STRING" not in raw_config
    assert "TELEGRAM_API_HASH" not in raw_config
    assert "BOT_TOKEN" not in raw_config


def test_gitignore_blocks_local_telethon_session_files() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "*.session" in ignored
    assert "*.session-journal" in ignored
