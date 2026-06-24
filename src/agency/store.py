"""Append-only stores for agency social permissions and controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.agency.models import SocialPermissionGrant, SocialPermissionStatus

if TYPE_CHECKING:
    from pathlib import Path


class AgencyControlState(BaseModel):
    """Global control switch for autonomous side effects."""

    emergency_stop: bool = False
    reason: str = ""


class FileSocialPermissionStore:
    """Small append-only JSONL store for social grants and control state."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._grants: dict[str, SocialPermissionGrant] = {}
        self._sent_events: list[dict[str, str]] = []
        self._control = AgencyControlState()
        self._load()

    def add(self, grant: SocialPermissionGrant) -> None:
        self._grants[grant.id] = grant
        self._append({"type": "grant_added", "grant": grant.model_dump(mode="json")})

    def list_grants(self) -> tuple[SocialPermissionGrant, ...]:
        return tuple(self._grants[key] for key in sorted(self._grants))

    def set_status(
        self,
        grant_id: str,
        status: SocialPermissionStatus,
    ) -> SocialPermissionGrant:
        grant = self._grants[grant_id].model_copy(update={"status": status})
        self._grants[grant_id] = grant
        self._append(
            {
                "type": "grant_status",
                "grant_id": grant_id,
                "status": status.value,
            }
        )
        return grant

    def pause(self, grant_id: str) -> SocialPermissionGrant:
        return self.set_status(grant_id, SocialPermissionStatus.PAUSED)

    def resume(self, grant_id: str) -> SocialPermissionGrant:
        return self.set_status(grant_id, SocialPermissionStatus.ACTIVE)

    def revoke(self, grant_id: str) -> SocialPermissionGrant:
        return self.set_status(grant_id, SocialPermissionStatus.REVOKED)

    def set_emergency_stop(self, enabled: bool, *, reason: str = "") -> None:
        self._control = AgencyControlState(emergency_stop=enabled, reason=reason)
        self._append(
            {
                "type": "control",
                "emergency_stop": enabled,
                "reason": reason,
            }
        )

    def emergency_stop_enabled(self) -> bool:
        return self._control.emergency_stop

    def control_state(self) -> AgencyControlState:
        return self._control

    def record_sent(
        self,
        *,
        grant_id: str,
        target_id: str,
        sent_at: datetime | None = None,
    ) -> None:
        """Record a completed social send for rate accounting.

        Gate evaluation never calls this method; the eventual send executor must
        record only after a real outbound send succeeds.
        """
        event = {
            "type": "social_send_recorded",
            "grant_id": grant_id,
            "target_id": target_id,
            "sent_at": (sent_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat(),
        }
        self._sent_events.append(
            {
                "grant_id": grant_id,
                "target_id": target_id,
                "sent_at": event["sent_at"],
            }
        )
        self._append(event)

    def count_sent_in_window(
        self,
        *,
        grant_id: str,
        now: datetime,
        window_seconds: int,
    ) -> int:
        current = now.astimezone(UTC)
        window = max(window_seconds, 0)
        count = 0
        for event in self._sent_events:
            if event["grant_id"] != grant_id:
                continue
            sent_at = _parse_datetime(event["sent_at"])
            if sent_at is None:
                continue
            age_seconds = (current - sent_at).total_seconds()
            if 0 <= age_seconds <= window:
                count += 1
        return count

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            self._apply_event(event)

    def _append(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _apply_event(self, event: dict[str, Any]) -> None:
        event_type: (
            Literal["grant_added", "grant_status", "control", "social_send_recorded"]
            | str
        ) = str(event.get("type", ""))
        if event_type == "grant_added":
            grant = SocialPermissionGrant.model_validate(event["grant"])
            self._grants[grant.id] = grant
            return
        if event_type == "grant_status":
            grant_id = str(event["grant_id"])
            if grant_id in self._grants:
                self._grants[grant_id] = self._grants[grant_id].model_copy(
                    update={
                        "status": SocialPermissionStatus(str(event["status"])),
                    }
                )
            return
        if event_type == "control":
            self._control = AgencyControlState(
                emergency_stop=bool(event.get("emergency_stop", False)),
                reason=str(event.get("reason", "")),
            )
            return
        if event_type == "social_send_recorded":
            self._sent_events.append(
                {
                    "grant_id": str(event.get("grant_id", "")),
                    "target_id": str(event.get("target_id", "")),
                    "sent_at": str(event.get("sent_at", "")),
                }
            )


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
