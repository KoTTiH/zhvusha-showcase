from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.mode_config import Mode


def should_update_personality(mode: Mode) -> bool:
    """Only personal mode interactions may modify personality files."""
    return mode == "personal"
