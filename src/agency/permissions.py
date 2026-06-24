"""Operator-facing control surface for social permission grants."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.agency.models import (  # noqa: TC001
    AgencyPermissionRequest,
    SocialPermissionGrant,
)
from src.agency.store import FileSocialPermissionStore  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable

PermissionControlAction = Literal["status", "pause", "resume", "revoke"]


class SocialPermissionControlResult(BaseModel):
    """Structured result for grant status/pause/resume/revoke commands."""

    action: PermissionControlAction
    ok: bool = True
    reason: str = ""
    message: str
    grants: tuple[SocialPermissionGrant, ...] = ()
    changed_grant: SocialPermissionGrant | None = None


class SocialPermissionController:
    """Thin controller over the append-only grant store.

    It only reports and changes grant lifecycle state. It does not execute
    Telegram/social actions and does not grant tool approvals.
    """

    def __init__(self, store: FileSocialPermissionStore) -> None:
        self._store = store

    def status(self, *, now: datetime | None = None) -> SocialPermissionControlResult:
        grants = self._store.list_grants()
        return SocialPermissionControlResult(
            action="status",
            message=render_social_permission_status(
                grants,
                emergency_stop=self._store.emergency_stop_enabled(),
                now=now,
            ),
            grants=grants,
        )

    def pause(self, grant_id: str) -> SocialPermissionControlResult:
        return self._change("pause", grant_id, self._store.pause)

    def resume(self, grant_id: str) -> SocialPermissionControlResult:
        return self._change("resume", grant_id, self._store.resume)

    def revoke(self, grant_id: str) -> SocialPermissionControlResult:
        return self._change("revoke", grant_id, self._store.revoke)

    def _change(
        self,
        action: PermissionControlAction,
        grant_id: str,
        operation: Callable[[str], SocialPermissionGrant],
    ) -> SocialPermissionControlResult:
        try:
            changed = operation(grant_id)
        except KeyError:
            return SocialPermissionControlResult(
                action=action,
                ok=False,
                reason="grant_not_found",
                message=f"Grant `{grant_id}` не найден.",
                grants=self._store.list_grants(),
            )
        return SocialPermissionControlResult(
            action=action,
            ok=True,
            message=f"Grant `{grant_id}` теперь `{changed.status.value}`.",
            grants=self._store.list_grants(),
            changed_grant=changed,
        )


def render_social_permission_status(
    grants: tuple[SocialPermissionGrant, ...],
    *,
    emergency_stop: bool = False,
    now: datetime | None = None,
) -> str:
    """Render a concise grant report for Жвуша/operator surfaces."""
    current = now or datetime.now(UTC)
    lines = ["Social permission grants:"]
    lines.append(f"emergency_stop: {'on' if emergency_stop else 'off'}")
    if not grants:
        lines.append("- нет активных или сохранённых grants.")
    for grant in grants:
        scopes = ", ".join(scope.value for scope in grant.scopes) or "none"
        topics = ", ".join(grant.allowed_topics) if grant.allowed_topics else "any"
        forbidden_topics = (
            ", ".join(grant.forbidden_topics) if grant.forbidden_topics else "none"
        )
        expires = (
            grant.expires_at.astimezone(UTC).isoformat()
            if grant.expires_at is not None
            else "never"
        )
        state = "usable" if grant.is_active(now=current) else "not usable"
        lines.append(
            "- "
            f"{grant.id}: {grant.target_id} [{grant.target_type.value}] "
            f"status={grant.status.value}, {state}, scopes={scopes}, "
            f"topics={topics}, forbidden_topics={forbidden_topics}, "
            f"privacy={grant.privacy_level}, "
            f"rate={grant.max_messages_per_window}/{grant.window_seconds}s, "
            f"expires={expires}"
        )
    lines.append(
        "Даже с grant Жвуша сохраняет право молчать, ждать, draft-ить "
        "или спросить Никиту; outbound действия требуют judgement/rate/privacy gates."
    )
    return "\n".join(lines)


def render_agency_permission_request(request: AgencyPermissionRequest) -> str:
    """Render a scoped permission ask without turning it into approval."""
    scopes = ", ".join(scope.value for scope in request.requested_scopes) or "none"
    duration = request.duration_hint or "not specified"
    return "\n".join(
        (
            "Запрос social permission:",
            f"target: {request.target_type.value} {request.target_id}",
            f"scopes: {scopes}",
            f"duration: {duration}",
            f"risk: {request.risk}",
            f"reason: {request.reason}",
            (
                "Это не является approval на отправку: даже после grant Жвуша "
                "сохраняет право молчать, wait, draft или спросить Никиту; "
                "outbound действия всё равно проходят judgement/rate/privacy gates."
            ),
        )
    )
