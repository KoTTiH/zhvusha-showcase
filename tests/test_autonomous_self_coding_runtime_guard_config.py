"""Configuration tests for autonomous self-coding runtime guard."""

from __future__ import annotations

import os
from unittest.mock import patch

from src.core.config import Settings

REQUIRED_ENV = {
    "BOT_TOKEN": "test_token",
    "CHANNEL_ID": "@test_channel",
    "ADMIN_USER_ID": "12345",
}


def _settings_no_env() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_autonomous_self_coding_runtime_guard_defaults_are_safe() -> None:
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        settings = _settings_no_env()

    assert settings.autonomous_self_coding_state_path == ""
    assert settings.autonomous_self_coding_restart_throttle_seconds == 21600
    assert settings.autonomous_self_coding_morning_guard_enabled is True
    assert settings.autonomous_self_coding_user_idle_seconds == 7200
    assert settings.autonomous_self_coding_user_activity_path == ""


def test_autonomous_self_coding_runtime_guard_settings_are_env_overridable() -> None:
    env = {
        **REQUIRED_ENV,
        "AUTONOMOUS_SELF_CODING_STATE_PATH": "/home/zhvusha/self-coding-state.json",
        "AUTONOMOUS_SELF_CODING_RESTART_THROTTLE_SECONDS": "900",
        "AUTONOMOUS_SELF_CODING_MORNING_GUARD_ENABLED": "false",
        "AUTONOMOUS_SELF_CODING_USER_IDLE_SECONDS": "3600",
        "AUTONOMOUS_SELF_CODING_USER_ACTIVITY_PATH": "/home/zhvusha/user-activity.json",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = _settings_no_env()

    assert settings.autonomous_self_coding_state_path == (
        "/home/zhvusha/self-coding-state.json"
    )
    assert settings.autonomous_self_coding_restart_throttle_seconds == 900
    assert settings.autonomous_self_coding_morning_guard_enabled is False
    assert settings.autonomous_self_coding_user_idle_seconds == 3600
    assert settings.autonomous_self_coding_user_activity_path == (
        "/home/zhvusha/user-activity.json"
    )
