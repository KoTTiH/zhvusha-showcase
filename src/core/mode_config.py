from __future__ import annotations

from typing import Literal

Mode = Literal["personal", "assistant", "social"]

MODE_ALLOWED_SKILLS: dict[Mode, list[str]] = {
    "personal": ["*"],
    "assistant": ["chat_response"],
    "social": ["chat_response"],
}

MODE_ALLOWED_CONTEXT: dict[Mode, list[str]] = {
    "personal": ["personality/*", "memory/people/*", "diary/*"],
    "assistant": ["personality/core.md", "personality/genes.md"],
    "social": ["personality/core.md", "personality/genes.md"],
}


def is_skill_allowed(skill_name: str, mode: Mode) -> bool:
    """Check if a skill is allowed to run in the given mode."""
    allowed = MODE_ALLOWED_SKILLS[mode]
    return "*" in allowed or skill_name in allowed
