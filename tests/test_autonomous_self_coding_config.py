"""Configuration tests for autonomous self-coding."""

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


def test_autonomous_self_coding_defaults_are_disabled() -> None:
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        settings = _settings_no_env()

    assert settings.autonomous_self_coding_enabled is False
    assert settings.autonomous_self_coding_interval_seconds == 21600
    assert settings.autonomous_self_coding_initial_delay_seconds == 300
    assert settings.autonomous_self_coding_max_tier == 3


def test_autonomous_self_coding_settings_are_env_overridable() -> None:
    env = {
        **REQUIRED_ENV,
        "AUTONOMOUS_SELF_CODING_ENABLED": "true",
        "AUTONOMOUS_SELF_CODING_INTERVAL_SECONDS": "900",
        "AUTONOMOUS_SELF_CODING_INITIAL_DELAY_SECONDS": "15",
        "AUTONOMOUS_SELF_CODING_MAX_TIER": "3",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = _settings_no_env()

    assert settings.autonomous_self_coding_enabled is True
    assert settings.autonomous_self_coding_interval_seconds == 900
    assert settings.autonomous_self_coding_initial_delay_seconds == 15
    assert settings.autonomous_self_coding_max_tier == 3
