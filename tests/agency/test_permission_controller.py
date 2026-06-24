"""Operator control surface for agency social permission grants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.agency.models import (
    AgencyPermissionRequest,
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialTargetType,
)
from src.agency.store import FileSocialPermissionStore


def _grant() -> SocialPermissionGrant:
    return SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(
            SocialPermissionScope.READ_ONLY,
            SocialPermissionScope.REPLY_IF_ADDRESSED,
        ),
        allowed_topics=("ZHVUSHA", "Telegram MCP"),
        forbidden_topics=("личные секреты",),
        max_messages_per_window=2,
        window_seconds=7200,
        expires_at=datetime(2026, 5, 14, tzinfo=UTC) + timedelta(hours=2),
    )


def test_permission_controller_status_pause_resume_revoke(tmp_path: Path) -> None:
    from src.agency.permissions import SocialPermissionController

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    store.add(_grant())
    controller = SocialPermissionController(store)

    status = controller.status(now=datetime(2026, 5, 14, tzinfo=UTC))
    assert status.action == "status"
    assert status.grants[0].id == "grant-devchat"
    assert "@devchat" in status.message
    assert "reply_if_addressed" in status.message
    assert "rate=2/7200s" in status.message
    assert "молчать" in status.message
    assert "forbidden_topics=личные секреты" in status.message

    paused = controller.pause("grant-devchat")
    assert paused.action == "pause"
    assert paused.changed_grant is not None
    assert paused.changed_grant.status == "paused"

    resumed = controller.resume("grant-devchat")
    assert resumed.action == "resume"
    assert resumed.changed_grant is not None
    assert resumed.changed_grant.status == "active"

    revoked = controller.revoke("grant-devchat")
    assert revoked.action == "revoke"
    assert revoked.changed_grant is not None
    assert revoked.changed_grant.status == "revoked"

    reloaded = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    assert reloaded.list_grants()[0].status == "revoked"


def test_permission_controller_unknown_grant_is_structured_error(
    tmp_path: Path,
) -> None:
    from src.agency.permissions import SocialPermissionController

    controller = SocialPermissionController(
        FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    )

    result = controller.pause("grant-missing")

    assert result.action == "pause"
    assert result.ok is False
    assert result.reason == "grant_not_found"
    assert "grant-missing" in result.message


def test_render_agency_permission_request_is_chat_ready_without_approval() -> None:
    from src.agency.permissions import render_agency_permission_request

    request = AgencyPermissionRequest(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        requested_scopes=(
            SocialPermissionScope.READ_ONLY,
            SocialPermissionScope.REPLY_IF_ADDRESSED,
        ),
        reason="Нужен контекст runtime обсуждения, но без автоспама.",
        risk="medium",
        duration_hint="до конца дня",
    )

    rendered = render_agency_permission_request(request)

    assert "Запрос social permission" in rendered
    assert "target: group @devchat" in rendered
    assert "scopes: read_only, reply_if_addressed" in rendered
    assert "duration: до конца дня" in rendered
    assert "risk: medium" in rendered
    assert "Нужен контекст runtime" in rendered
    assert "не является approval на отправку" in rendered
    assert "молчать" in rendered
    assert "draft" in rendered
    assert "спросить Никиту" in rendered
