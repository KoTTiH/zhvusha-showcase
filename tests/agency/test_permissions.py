from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.agency.models import (
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialPermissionStatus,
    SocialTargetType,
)
from src.agency.store import FileSocialPermissionStore


def test_permission_grant_scope_pause_resume_revoke_and_expiry() -> None:
    now = datetime(2026, 5, 13, tzinfo=UTC)
    grant = SocialPermissionGrant(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.READ_ONLY,),
        expires_at=now + timedelta(minutes=30),
    )

    assert grant.permits(SocialPermissionScope.READ_ONLY, now=now)
    assert not grant.permits(SocialPermissionScope.REPLY_IF_ADDRESSED, now=now)
    assert not grant.model_copy(
        update={"status": SocialPermissionStatus.PAUSED}
    ).permits(SocialPermissionScope.READ_ONLY, now=now)
    assert grant.model_copy(update={"status": SocialPermissionStatus.ACTIVE}).permits(
        SocialPermissionScope.READ_ONLY, now=now
    )
    assert not grant.model_copy(
        update={"status": SocialPermissionStatus.REVOKED}
    ).permits(SocialPermissionScope.READ_ONLY, now=now)
    assert not grant.permits(
        SocialPermissionScope.READ_ONLY,
        now=now + timedelta(hours=1),
    )


def test_file_store_persists_grants_and_emergency_stop(tmp_path: Path) -> None:
    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = SocialPermissionGrant(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.READ_ONLY,),
    )

    store.add(grant)
    store.set_emergency_stop(True, reason="manual stop")

    reloaded = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    assert [item.id for item in reloaded.list_grants()] == [grant.id]
    assert reloaded.emergency_stop_enabled() is True
    assert reloaded.control_state().reason == "manual stop"


def test_file_store_counts_sent_messages_inside_grant_window(tmp_path: Path) -> None:
    now = datetime(2026, 5, 14, 12, tzinfo=UTC)
    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        window_seconds=3600,
    )
    store.add(grant)

    store.record_sent(
        grant_id=grant.id,
        target_id="@devchat",
        sent_at=now - timedelta(minutes=30),
    )
    store.record_sent(
        grant_id=grant.id,
        target_id="@devchat",
        sent_at=now - timedelta(hours=2),
    )

    reloaded = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    assert (
        reloaded.count_sent_in_window(
            grant_id=grant.id,
            now=now,
            window_seconds=grant.window_seconds,
        )
        == 1
    )
