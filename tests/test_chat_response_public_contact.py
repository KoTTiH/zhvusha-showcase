"""Public Telegram handle from settings must reach the assistant/social
prompt when configured, and stay absent when not.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"


def _settings(contact: str = "@nikita_dev") -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path="unused_ws",
        claude_cli_path="claude",
        public_info_about_nikita="Никита — разработчик.",
        public_contact_nikita=contact,
        admin_user_id=42,
        chat_assistant_tier="analyst",
    )


def _assistant_system(contact: str = "@nikita_dev", interaction_count: int = 5) -> str:
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(contact=contact)):
        return skill._build_system(
            "assistant",
            personality_context="",
            public_info="Никита — разработчик.",
            interaction_count=interaction_count,
            people_context="",
            current_user_id=999,
        )


def test_contact_visible_in_assistant_prompt_when_set() -> None:
    assert "@nikita_dev" in _assistant_system(contact="@nikita_dev")


def test_contact_visible_in_intro_prompt_when_set() -> None:
    assert "@nikita_dev" in _assistant_system(
        contact="@nikita_dev", interaction_count=1
    )


def test_contact_absent_when_setting_empty() -> None:
    system = _assistant_system(contact="")
    assert "{public_contact" not in system  # no unfilled placeholder leaks
