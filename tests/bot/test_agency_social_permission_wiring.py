"""Production wiring for agency social permission state."""

from __future__ import annotations

from pathlib import Path

from src.agency.models import (
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialPermissionStatus,
    SocialTargetType,
)
from src.core.config import Settings
from src.skills.base import AgentContext


def test_social_permission_store_uses_configured_path(tmp_path: Path) -> None:
    from src.agency.store import FileSocialPermissionStore
    from src.bot.main import _build_social_permission_store

    permissions_path = tmp_path / "agency" / "permissions.jsonl"
    settings = Settings(
        bot_token="token",
        channel_id="1",
        admin_user_id=123,
        agency_permission_store_path=permissions_path.as_posix(),
    )

    store = _build_social_permission_store(settings)
    store.add(
        SocialPermissionGrant(
            id="grant-devchat",
            target_id="@devchat",
            target_type=SocialTargetType.GROUP,
            scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        )
    )

    reloaded = FileSocialPermissionStore(permissions_path)
    assert permissions_path.exists()
    assert reloaded.list_grants()[0].id == "grant-devchat"


def test_social_permission_command_status_and_pause(tmp_path: Path) -> None:
    from src.agency.store import FileSocialPermissionStore
    from src.bot.main import _social_permission_control_reply

    store_path = tmp_path / "agency" / "permissions.jsonl"
    store = FileSocialPermissionStore(store_path)
    store.add(
        SocialPermissionGrant(
            id="grant-devchat",
            target_id="@devchat",
            target_type=SocialTargetType.GROUP,
            scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        )
    )
    context = AgentContext(user_id=123, chat_id=123, mode="personal")

    status = _social_permission_control_reply(
        "/social_permissions status",
        context,
        admin_user_id=123,
        store=store,
    )
    paused = _social_permission_control_reply(
        "/social_permissions pause grant-devchat",
        context,
        admin_user_id=123,
        store=store,
    )

    assert status is not None
    assert "Social permission grants:" in status
    assert "@devchat" in status
    assert paused == "Grant `grant-devchat` теперь `paused`."
    reloaded = FileSocialPermissionStore(store_path)
    assert reloaded.list_grants()[0].status is SocialPermissionStatus.PAUSED


def test_social_permission_command_does_not_leak_to_non_admin(tmp_path: Path) -> None:
    from src.agency.store import FileSocialPermissionStore
    from src.bot.main import _social_permission_control_reply

    store = FileSocialPermissionStore(tmp_path / "agency" / "permissions.jsonl")
    store.add(
        SocialPermissionGrant(
            id="grant-devchat",
            target_id="@devchat",
            target_type=SocialTargetType.GROUP,
            scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        )
    )
    context = AgentContext(user_id=456, chat_id=456, mode="assistant")

    reply = _social_permission_control_reply(
        "/social_permissions status",
        context,
        admin_user_id=123,
        store=store,
    )

    assert reply == "Эта команда доступна только Никите."
