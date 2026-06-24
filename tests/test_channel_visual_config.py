"""Channel visual configuration defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import Settings

if TYPE_CHECKING:
    import pytest


def test_channel_visual_image_generation_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IMAGE_GENERATION_ENABLED", raising=False)

    settings = Settings(
        bot_token="fake",
        channel_id="@test",
        admin_user_id=1,
        _env_file=None,
    )

    assert settings.image_generation_enabled is False
    assert settings.image_generation_provider == "cli"
    assert settings.image_generation_model == ""
    assert settings.image_generation_size == "1024x1024"
    assert settings.image_generation_cli_command == ""
    assert settings.image_generation_cli_timeout_seconds == 300.0


def test_openai_api_key_is_masked_in_settings_repr() -> None:
    settings = Settings(
        bot_token="fake",
        channel_id="@test",
        admin_user_id=1,
        openai_api_key="fake_openai_secret",
    )

    rendered = repr(settings)

    assert "fake_openai_secret" not in rendered
    assert "openai_api_key=sk-o***" in rendered
