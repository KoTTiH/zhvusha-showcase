"""Standalone daemon wiring for live process ownership."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_standalone_daemon_ownership_blocks_second_live_owner(
    tmp_path: Path,
) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard
    from src.daemon.main import _acquire_standalone_daemon_ownership

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid in {111, 222},
    )
    _acquire_standalone_daemon_ownership(
        workspace_root=tmp_path,
        owner_id="daemon-a",
        pid=111,
        guard=guard,
    )

    with pytest.raises(RuntimeError, match="daemon: already_owned"):
        _acquire_standalone_daemon_ownership(
            workspace_root=tmp_path,
            owner_id="daemon-b",
            pid=222,
            guard=guard,
        )

    owner = guard.status("daemon").owner
    assert owner is not None
    assert owner.owner_id == "daemon-a"


async def test_standalone_daemon_ownership_release_clears_owner(
    tmp_path: Path,
) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard
    from src.daemon.main import (
        _acquire_standalone_daemon_ownership,
        _release_standalone_daemon_ownership,
    )

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 222,
    )
    ownership = _acquire_standalone_daemon_ownership(
        workspace_root=tmp_path,
        owner_id="daemon-b",
        pid=222,
        guard=guard,
    )

    await _release_standalone_daemon_ownership(ownership)

    assert guard.status("daemon").reason == "not_owned"
