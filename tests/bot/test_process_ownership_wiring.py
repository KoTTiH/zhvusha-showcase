"""Bot composition wiring for live process ownership."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_live_process_services_include_enabled_components() -> None:
    from src.bot.main import _live_process_services

    assert _live_process_services(
        daemon_enabled=True,
        telegram_mcp_enabled=True,
        personal_telegram_inbound_enabled=True,
    ) == ("bot", "daemon", "telegram_mcp", "telegram_inbound")
    assert _live_process_services(
        daemon_enabled=False,
        telegram_mcp_enabled=False,
        personal_telegram_inbound_enabled=False,
    ) == ("bot",)


def test_live_process_ownership_blocks_second_live_owner(tmp_path: Path) -> None:
    from src.bot.main import (
        _acquire_live_process_ownership,
        _LiveProcessOwnershipConflictError,
    )
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid in {111, 222},
    )
    _acquire_live_process_ownership(
        workspace_root=tmp_path,
        services=("bot",),
        owner_id="session-a",
        pid=111,
        guard=guard,
    )

    with pytest.raises(_LiveProcessOwnershipConflictError, match="bot: already_owned"):
        _acquire_live_process_ownership(
            workspace_root=tmp_path,
            services=("bot",),
            owner_id="session-b",
            pid=222,
            guard=guard,
        )

    owner = guard.status("bot").owner
    assert owner is not None
    assert owner.owner_id == "session-a"


def test_live_process_ownership_releases_partial_acquire_on_failure(
    tmp_path: Path,
) -> None:
    from src.bot.main import (
        _acquire_live_process_ownership,
        _LiveProcessOwnershipConflictError,
    )
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid in {111, 222},
    )
    guard.acquire(service="daemon", owner_id="session-a", pid=111)

    with pytest.raises(
        _LiveProcessOwnershipConflictError, match="daemon: already_owned"
    ):
        _acquire_live_process_ownership(
            workspace_root=tmp_path,
            services=("bot", "daemon"),
            owner_id="session-b",
            pid=222,
            guard=guard,
        )

    assert guard.status("bot").reason == "not_owned"
    daemon_owner = guard.status("daemon").owner
    assert daemon_owner is not None
    assert daemon_owner.owner_id == "session-a"


async def test_release_live_process_ownership_clears_owned_services(
    tmp_path: Path,
) -> None:
    from src.bot.main import (
        _acquire_live_process_ownership,
        _release_live_process_ownership,
    )
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 222,
    )
    ownership = _acquire_live_process_ownership(
        workspace_root=tmp_path,
        services=("bot", "telegram_mcp"),
        owner_id="session-b",
        pid=222,
        guard=guard,
    )

    await _release_live_process_ownership(ownership)

    assert guard.status("bot").reason == "not_owned"
    assert guard.status("telegram_mcp").reason == "not_owned"


async def test_close_live_startup_resources_closes_unique_clients_once() -> None:
    from src.bot.main import _close_live_startup_resources

    class _RedisClient:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    class _Engine:
        def __init__(self) -> None:
            self.dispose_count = 0

        async def dispose(self) -> None:
            self.dispose_count += 1

    class _Session:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1

    class _Bot:
        def __init__(self) -> None:
            self.session = _Session()

    redis = _RedisClient()
    engine = _Engine()
    bot = _Bot()

    await _close_live_startup_resources(
        bot=bot,  # type: ignore[arg-type]
        redis_clients=(redis, redis),
        db_engine=engine,
    )

    assert redis.close_count == 1
    assert engine.dispose_count == 1
    assert bot.session.close_count == 1


def test_live_process_status_reply_renders_owner_report(tmp_path: Path) -> None:
    from src.bot.main import _live_process_status_reply
    from src.core.process_guard import FileProcessOwnershipGuard
    from src.skills.base import AgentContext

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 111,
    )
    guard.acquire(service="bot", owner_id="session-a", pid=111)
    context = AgentContext(user_id=123, chat_id=123, mode="personal")

    reply = _live_process_status_reply(
        "/process_status",
        context,
        admin_user_id=123,
        workspace_root=tmp_path,
        guard=guard,
    )

    assert reply is not None
    assert "Process ownership:" in reply
    assert "bot: owned" in reply
    assert "owner=session-a" in reply
    assert "daemon: not_owned" in reply
    assert "telegram_inbound: not_owned" in reply


def test_live_process_status_reply_does_not_leak_to_non_admin(
    tmp_path: Path,
) -> None:
    from src.bot.main import _live_process_status_reply
    from src.core.process_guard import FileProcessOwnershipGuard
    from src.skills.base import AgentContext

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 111,
    )
    guard.acquire(service="bot", owner_id="session-a", pid=111)
    context = AgentContext(user_id=456, chat_id=456, mode="assistant")

    reply = _live_process_status_reply(
        "/process_status",
        context,
        admin_user_id=123,
        workspace_root=tmp_path,
        guard=guard,
    )

    assert reply == "Эта команда доступна только Никите."
