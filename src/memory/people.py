from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from src.core.config import get_settings
from src.memory.protocols import PeopleManagerProtocol, PersonProfile

if TYPE_CHECKING:
    from src.core.mode_config import Mode

logger = structlog.get_logger()


class PeopleManager(PeopleManagerProtocol):
    """File-based people profile storage.

    Profiles are stored as markdown files with simple key-value frontmatter.
    Path: {workspace}/memory/people/{user_id}/profile.md

    Implements :class:`PeopleManagerProtocol`. Returns :class:`PersonProfile`
    TypedDict instances (which are just ``dict[str, Any]`` at runtime).
    """

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root / "memory" / "people"

    def get_or_create_profile(
        self,
        user_id: int,
        username: str = "",
        first_name: str = "",
    ) -> PersonProfile:
        """Get existing profile or create a new one."""
        profile_dir = self._root / str(user_id)
        profile_file = profile_dir / "profile.md"

        if profile_file.exists():
            return cast("PersonProfile", self._parse_profile(profile_file))

        profile_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(tz=UTC).isoformat()
        profile: PersonProfile = {
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "significance": "stranger",
            "interaction_count": 0,
            "first_seen": now,
            "last_seen": now,
        }
        self._write_profile(profile_file, dict(profile))
        logger.info("people_profile_created", user_id=user_id)
        return profile

    def update_profile(self, user_id: int, updates: dict[str, Any]) -> None:
        """Update profile fields."""
        profile_file = self._root / str(user_id) / "profile.md"
        if not profile_file.exists():
            return
        profile = self._parse_profile(profile_file)
        profile.update(updates)
        self._write_profile(profile_file, profile)

    _PROMOTE_THRESHOLD: int = 3

    def record_interaction(self, user_id: int) -> bool:
        """Increment interaction counter, update last_seen, auto-promote.

        Returns True if the user was promoted from stranger to known.
        """
        profile_file = self._root / str(user_id) / "profile.md"
        if not profile_file.exists():
            return False
        profile = self._parse_profile(profile_file)
        profile["interaction_count"] = int(profile.get("interaction_count", 0)) + 1
        profile["last_seen"] = datetime.now(tz=UTC).isoformat()

        promoted = False
        # Auto-promote stranger → known at threshold
        if (
            profile.get("significance") == "stranger"
            and profile["interaction_count"] >= self._PROMOTE_THRESHOLD
        ):
            profile["significance"] = "known"
            promoted = True
            logger.info(
                "people_auto_promoted",
                user_id=user_id,
                interaction_count=profile["interaction_count"],
            )

        self._write_profile(profile_file, profile)
        return promoted

    def get_interaction_count(self, user_id: int) -> int:
        """Return interaction count or 0 if profile doesn't exist."""
        profile_file = self._root / str(user_id) / "profile.md"
        if not profile_file.exists():
            return 0
        profile = self._parse_profile(profile_file)
        return int(profile.get("interaction_count", 0))

    def get_significance_level(self, user_id: int) -> str:
        """Return significance level or 'stranger' if profile doesn't exist."""
        profile_file = self._root / str(user_id) / "profile.md"
        if not profile_file.exists():
            return "stranger"
        profile = self._parse_profile(profile_file)
        result: str = profile.get("significance", "stranger")
        return result

    def get_profile_for_context(self, user_id: int, mode: Mode) -> str:
        """Return profile content appropriate for the mode.

        Personal: full profile text.
        Assistant: this person's profile only.
        Social: empty (no personal data).
        """
        if mode == "social":
            return ""
        profile_file = self._root / str(user_id) / "profile.md"
        if not profile_file.exists():
            return ""
        return profile_file.read_text(encoding="utf-8")

    def _parse_profile(self, path: Path) -> dict[str, Any]:
        """Parse simple frontmatter (key: value between --- markers)."""
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")

        profile: dict[str, Any] = {}
        in_frontmatter = False

        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter and ": " in stripped:
                key, value = stripped.split(": ", 1)
                # Parse numeric values
                try:
                    profile[key] = int(value)
                except ValueError:
                    profile[key] = value

        return profile

    def _write_profile(self, path: Path, profile: dict[str, Any]) -> None:
        """Write profile as markdown with frontmatter."""
        lines = ["---"]
        for key, value in profile.items():
            lines.append(f"{key}: {value}")
        lines.append("---")
        lines.append("")
        lines.append("# Notes")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


_manager: PeopleManager | None = None


def get_people_manager() -> PeopleManager:
    """Get or create the singleton PeopleManager."""
    global _manager
    if _manager is None:
        settings = get_settings()
        root = Path(settings.workspace_path).expanduser().resolve()
        _manager = PeopleManager(root)
    return _manager
