"""Tests for ConsolidationLock."""

from __future__ import annotations

import os

from src.memory.consolidation_lock import ConsolidationLock


async def test_acquire_when_no_lock(tmp_path):
    lock = ConsolidationLock(tmp_path)
    assert await lock.try_acquire() is True
    assert lock.lock_path.exists()
    assert lock.lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())


async def test_acquire_blocked_by_live_process(tmp_path):
    lock = ConsolidationLock(tmp_path)
    # Write current PID (alive process)
    lock.lock_path.write_text(str(os.getpid()), encoding="utf-8")

    # Another "process" trying to acquire — same PID means it's us
    # but the lock file already exists with our PID from a fresh write
    lock2 = ConsolidationLock(tmp_path)
    # Since the PID is alive and lock is fresh, it should be blocked
    result = await lock2.try_acquire()
    # Actually it will succeed because our PID re-writes it
    # The real test: write a different alive PID
    # Use PID 1 (init, always alive on Linux)
    lock.lock_path.write_text("1", encoding="utf-8")
    lock3 = ConsolidationLock(tmp_path)
    result = await lock3.try_acquire()
    assert result is False


async def test_acquire_reclaims_stale_lock(tmp_path):
    lock = ConsolidationLock(tmp_path)
    # Write PID 1 and backdate mtime
    lock.lock_path.write_text("1", encoding="utf-8")
    # Make it look old (> 1 hour)
    old_time = lock.lock_path.stat().st_mtime - 7200
    os.utime(str(lock.lock_path), (old_time, old_time))

    result = await lock.try_acquire()
    assert result is True


async def test_release_clears_lock(tmp_path):
    lock = ConsolidationLock(tmp_path)
    await lock.try_acquire()
    assert lock.lock_path.exists()

    await lock.release()
    assert not lock.lock_path.exists()


async def test_last_consolidated_at_no_file(tmp_path):
    lock = ConsolidationLock(tmp_path)
    assert await lock.read_last_consolidated_at() == 0.0


async def test_last_consolidated_at_with_file(tmp_path):
    lock = ConsolidationLock(tmp_path)
    await lock.mark_consolidated_at(1234.5)
    result = await lock.read_last_consolidated_at()
    assert result == 1234.5


async def test_release_keeps_last_consolidated_marker(tmp_path):
    lock = ConsolidationLock(tmp_path)
    await lock.try_acquire()
    await lock.mark_consolidated_at(1234.5)

    await lock.release()

    assert not lock.lock_path.exists()
    assert await lock.read_last_consolidated_at() == 1234.5
