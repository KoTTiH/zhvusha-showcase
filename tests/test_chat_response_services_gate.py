"""Assistant prompts must not proactively pitch the creator's services.
We only verify structural invariants here — no matching against specific
phrasings, so the prompt can evolve without rewriting these tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path="unused_ws",
        claude_cli_path="claude",
        public_info_about_nikita="NIKITA_PUBLIC_INFO_MARKER",
        public_contact_nikita="",
        admin_user_id=42,
        chat_assistant_tier="analyst",
    )


def _build(interaction_count: int, mode: str = "assistant") -> str:
    skill = ChatResponseSkill()
    with patch(_PATCH_SETTINGS, return_value=_settings()):
        return skill._build_system(
            mode,  # type: ignore[arg-type]
            personality_context="P",
            public_info="NIKITA_PUBLIC_INFO_MARKER",
            interaction_count=interaction_count,
            people_context="",
            current_user_id=999 if mode != "personal" else 42,
        )


def test_intro_does_not_carry_services_menu() -> None:
    system = _build(interaction_count=1)
    # The old menu option explicitly pitched services in the intro — must
    # be gone. The check is intentionally loose: any leftover '(разработка'
    # inline in the intro would flag a menu-style pitch.
    assert "Услуги Никиты (разработка" not in system


def test_assistant_does_not_carry_unconditional_services_block() -> None:
    system = _build(interaction_count=5)
    assert "## Информация об услугах Никиты" not in system


def test_public_info_still_reachable_in_assistant() -> None:
    """Public info isn't forbidden — it's only *gated*. It must still
    appear in the prompt so the model can reach for it on request."""
    system = _build(interaction_count=5)
    assert "NIKITA_PUBLIC_INFO_MARKER" in system


def test_personal_system_has_no_services_block_at_all() -> None:
    """Creator already knows his own services — no conditional block needed."""
    system = _build(interaction_count=5, mode="personal")
    assert "NIKITA_PUBLIC_INFO_MARKER" not in system
