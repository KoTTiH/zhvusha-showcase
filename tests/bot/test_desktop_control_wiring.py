"""Bot startup wiring for Desktop Control runtime surface."""

from __future__ import annotations

from pathlib import Path

from src.core.config import Settings


def test_desktop_control_runtime_surface_is_disabled_without_flag(
    tmp_path: Path,
) -> None:
    from src.bot.main import _build_desktop_control_runtime_surface

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        desktop_control_enabled=False,
        desktop_control_command_map_json=(
            '{"desktop.media_control:pause":["playerctl","pause"]}'
        ),
    )

    assert (
        _build_desktop_control_runtime_surface(settings, workspace_root=tmp_path)
        is None
    )


def test_desktop_control_runtime_surface_registers_allowlisted_tools(
    tmp_path: Path,
) -> None:
    from src.bot.main import _build_desktop_control_runtime_surface

    calls: list[tuple[str, ...]] = []

    async def runner(argv: tuple[str, ...]) -> str:
        calls.append(argv)
        return "ok"

    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        desktop_control_enabled=True,
        desktop_control_command_map_json=(
            '{"desktop.media_control:pause":["playerctl","pause"],'
            '"desktop.app_launcher:telegram":["gtk-launch","org.telegram.desktop.desktop"]}'
        ),
    )

    surface = _build_desktop_control_runtime_surface(
        settings,
        workspace_root=tmp_path,
        command_runner=runner,
    )

    assert surface is not None
    tools = {tool.name: tool.capability for tool in surface.gateway.registered_tools()}
    assert tools == {
        "desktop_app_launcher": "desktop_app_launcher",
        "desktop_media_control": "desktop_media_control",
    }
    assert surface.worker.name == "desktop_control"
    assert (tmp_path / "agent_runtime" / "desktop_control_audit.jsonl").parent.exists()
