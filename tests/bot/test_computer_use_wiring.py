"""Bot wiring for computer-use runtime surface without slash-command sprawl."""

from __future__ import annotations

from pathlib import Path

from src.core.config import Settings


def test_computer_use_runtime_surface_is_disabled_without_flag(tmp_path: Path) -> None:
    from src.bot.main import _build_computer_use_runtime_surface

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        computer_use_enabled=False,
        live_browser_control_enabled=True,
    )

    assert (
        _build_computer_use_runtime_surface(settings, workspace_root=tmp_path) is None
    )


def test_computer_use_runtime_surface_registers_live_browser_tools(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.computer_use import LocalChromeDevToolsClient
    from src.bot.main import _build_computer_use_runtime_surface

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        computer_use_enabled=True,
        live_browser_control_enabled=True,
        live_browser_backend="chrome_devtools_mcp",
        live_browser_debug_url="http://127.0.0.1:9222",
        live_browser_auto_launch=False,
    )

    surface = _build_computer_use_runtime_surface(settings, workspace_root=tmp_path)

    assert surface is not None
    tools = {tool.name: tool.capability for tool in surface.gateway.registered_tools()}
    assert tools["browser_live_status"] == "browser_live_control"
    assert tools["browser_live_navigate"] == "browser_navigate"
    assert tools["browser_live_interactive_task"] == "browser_interactive_task"
    assert tools["computer_browser_submit"] == "browser_submit"
    browser_tool = next(
        tool
        for tool in surface.gateway.registered_tools()
        if tool.name == "browser_live_navigate"
    )
    assert isinstance(
        browser_tool._live_browser_adapter._cdp_client,
        LocalChromeDevToolsClient,
    )
    assert surface.worker.name == "computer_use"


def test_computer_use_runtime_surface_can_auto_launch_live_browser(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.computer_use import ManagedChromeDevToolsClient
    from src.bot.main import _build_computer_use_runtime_surface

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        computer_use_enabled=True,
        live_browser_control_enabled=True,
        live_browser_backend="chrome_devtools_mcp",
        live_browser_debug_url="http://127.0.0.1:9222",
        live_browser_auto_launch=True,
        live_browser_executable="/usr/bin/chromium",
        live_browser_user_data_dir=str(tmp_path / "profile"),
        live_browser_headless=False,
        browser_http_proxy="http://127.0.0.1:7897",
    )

    surface = _build_computer_use_runtime_surface(settings, workspace_root=tmp_path)

    assert surface is not None
    browser_tool = next(
        tool
        for tool in surface.gateway.registered_tools()
        if tool.name == "browser_live_navigate"
    )
    client = browser_tool._live_browser_adapter._cdp_client
    assert isinstance(client, ManagedChromeDevToolsClient)
    assert client._proxy == "http://127.0.0.1:7897"
    assert client._headless is False


def test_computer_use_runtime_surface_registers_desktop_tools_without_live_browser(
    tmp_path: Path,
) -> None:
    from src.bot.main import _build_computer_use_runtime_surface

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        computer_use_enabled=True,
        live_browser_control_enabled=False,
    )

    surface = _build_computer_use_runtime_surface(settings, workspace_root=tmp_path)

    assert surface is not None
    tools = {tool.name: tool.capability for tool in surface.gateway.registered_tools()}
    assert tools["desktop_screenshot"] == "desktop_screenshot"
    assert tools["desktop_hotkeys"] == "desktop_hotkeys"
    assert "browser_live_status" not in tools
    assert surface.worker.name == "computer_use"


def test_computer_use_does_not_add_user_facing_slash_commands() -> None:
    from src.bot.main import _ADMIN_BOT_COMMAND_SPECS

    command_names = {name for name, _description in _ADMIN_BOT_COMMAND_SPECS}

    assert "computer_status" not in command_names
    assert "computer_pause" not in command_names
    assert "computer_resume" not in command_names
    assert "computer_stop" not in command_names
