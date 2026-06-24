"""System prompt must carry a structured IDENTITY block so the LLM grounds
its reasoning on user_id, not on stylistic cues from the conversation.

Root cause of the real incident in
``workspace/logs/8625506318/chat_2026-04-10.jsonl``: Zhvusha read "KoTTiHy"
from Telegram metadata, picked up "Никита" from core.md, and concluded she
was talking to her creator. The gate below forces the LLM to compare
user_ids explicitly for any identity assertion.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"


def _settings(admin_user_id: int = 42) -> SimpleNamespace:
    # workspace_path is unused here — _build_system never touches disk.
    return SimpleNamespace(
        workspace_path="unused_ws",
        claude_cli_path="claude",
        public_info_about_nikita="Никита — разработчик.",
        admin_user_id=admin_user_id,
        chat_assistant_tier="analyst",
    )


def test_personal_mode_marks_user_as_creator() -> None:
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "personal",
            personality_context="P",
            public_info="info",
            interaction_count=5,
            people_context="",
            current_user_id=42,
        )
    assert "<IDENTITY>" in system
    assert "creator_user_id: 42" in system
    assert "current_user_id: 42" in system
    assert "is_creator: true" in system


def test_assistant_mode_marks_user_as_non_creator() -> None:
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "assistant",
            personality_context="P",
            public_info="info",
            interaction_count=5,
            people_context="",
            current_user_id=999,
        )
    assert "<IDENTITY>" in system
    assert "creator_user_id: 42" in system
    assert "current_user_id: 999" in system
    assert "is_creator: false" in system


def test_assistant_mode_includes_identity_rules() -> None:
    """Rules that tell the LLM: user_id is the only identity signal."""
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "assistant",
            personality_context="P",
            public_info="info",
            interaction_count=5,
            people_context="",
            current_user_id=999,
        )
    # Rules must mention the user_id-only gate and forbid the intimate tokens
    assert "user_id" in system.lower()
    assert "Никита" in system
    assert "ребёночек" in system
    # Signals to ignore
    assert "стил" in system.lower() or "имя" in system.lower()


def test_social_mode_includes_identity_rules() -> None:
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "social",
            personality_context="P",
            public_info="info",
            interaction_count=0,
            people_context="",
            current_user_id=999,
        )
    assert "<IDENTITY>" in system
    assert "is_creator: false" in system
    assert "user_id" in system.lower()


def test_assistant_intro_mode_also_has_identity_block() -> None:
    """Intro (interaction_count ≤ 2) uses a different template; identity
    must land there too — first contact is precisely when confusion starts."""
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "assistant",
            personality_context="",
            public_info="info",
            interaction_count=1,
            people_context="",
            current_user_id=999,
        )
    assert "<IDENTITY>" in system
    assert "is_creator: false" in system


def test_personal_mode_skips_long_rules_to_preserve_voice() -> None:
    """In personal mode the short IDENTITY block alone is enough — full
    rules would over-formalize Zhvusha's voice with her creator."""
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings(admin_user_id=42)):
        system = skill._build_system(
            "personal",
            personality_context="P",
            public_info="info",
            interaction_count=5,
            people_context="",
            current_user_id=42,
        )
    assert "<IDENTITY>" in system
    # Long non-personal rule phrases must not leak into personal mode
    assert "определяется ИСКЛЮЧИТЕЛЬНО по user_id" not in system
