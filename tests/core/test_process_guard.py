"""Live runtime process ownership guard contract."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any


def _acquire_process_guard_worker(
    path_text: str,
    start_event: Any,
    result_queue: Any,
    owner_id: str,
    pid: int,
) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard

    start_event.wait(5)
    guard = FileProcessOwnershipGuard(
        Path(path_text),
        pid_is_alive=lambda _pid: True,
    )
    status = guard.acquire(service="bot", owner_id=owner_id, pid=pid)
    result_queue.put(
        (
            status.acquired,
            status.reason,
            status.owner.owner_id if status.owner is not None else "",
        )
    )


def test_process_guard_blocks_second_live_owner(tmp_path: Path) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 111,
    )

    first = guard.acquire(service="bot", owner_id="session-a", pid=111)
    second = guard.acquire(service="bot", owner_id="session-b", pid=222)

    assert first.acquired is True
    assert second.acquired is False
    assert second.reason == "already_owned"
    assert second.owner is not None
    assert second.owner.pid == 111
    assert guard.status("bot").owner is not None


def test_process_guard_replaces_stale_owner(tmp_path: Path) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard

    path = tmp_path / "owners.json"
    first = FileProcessOwnershipGuard(path, pid_is_alive=lambda pid: pid == 111)
    first.acquire(service="daemon", owner_id="old", pid=111)

    second = FileProcessOwnershipGuard(path, pid_is_alive=lambda pid: pid == 222)
    acquired = second.acquire(service="daemon", owner_id="new", pid=222)

    assert acquired.acquired is True
    assert acquired.reason == "stale_replaced"
    assert acquired.owner is not None
    assert acquired.owner.owner_id == "new"


def test_process_guard_renders_operator_status_report(tmp_path: Path) -> None:
    from src.core.process_guard import (
        FileProcessOwnershipGuard,
        render_process_ownership_report,
    )

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 111,
    )
    guard.acquire(service="bot", owner_id="session-a", pid=111)
    guard.acquire(service="daemon", owner_id="old-daemon", pid=333)

    report = render_process_ownership_report(
        (
            guard.status("bot"),
            guard.status("daemon"),
            guard.status("telegram_mcp"),
        )
    )

    assert "bot: owned" in report
    assert "pid=111" in report
    assert "owner=session-a" in report
    assert "daemon: stale" in report
    assert "telegram_mcp: not_owned" in report


def test_process_guard_heartbeat_updates_current_owner(tmp_path: Path) -> None:
    from src.core.process_guard import FileProcessOwnershipGuard

    guard = FileProcessOwnershipGuard(
        tmp_path / "owners.json",
        pid_is_alive=lambda pid: pid == 111,
    )
    acquired = guard.acquire(service="bot", owner_id="session-a", pid=111)

    heartbeat = guard.heartbeat("bot", owner_id="session-a")
    mismatch = guard.heartbeat("bot", owner_id="other")

    assert acquired.owner is not None
    assert heartbeat.acquired is True
    assert heartbeat.reason == "heartbeat"
    assert heartbeat.owner is not None
    assert heartbeat.owner.heartbeat_at >= acquired.owner.heartbeat_at
    assert mismatch.acquired is False
    assert mismatch.reason == "owner_mismatch"


def test_process_guard_acquire_is_atomic_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "owners.json"
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_acquire_process_guard_worker,
            args=(
                str(path),
                start_event,
                result_queue,
                f"session-{index}",
                9000 + index,
            ),
        )
        for index in range(4)
    ]

    for process in processes:
        process.start()
    start_event.set()

    results: list[tuple[bool, str, str]] = []
    try:
        for _process in processes:
            try:
                results.append(result_queue.get(timeout=10))
            except Empty as exc:
                raise AssertionError("process guard worker did not return") from exc
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sum(1 for acquired, _reason, _owner in results if acquired) == 1
    assert sum(1 for acquired, _reason, _owner in results if not acquired) == 3
    assert {reason for _acquired, reason, _owner in results} == {
        "acquired",
        "already_owned",
    }
