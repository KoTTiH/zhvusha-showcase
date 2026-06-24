"""File-based lock to prevent concurrent consolidation runs."""

from __future__ import annotations

import os
import time
from pathlib import Path  # noqa: TC003  # used at runtime in __init__

import structlog

logger = structlog.get_logger()

_STALE_SECONDS = 3600  # 1 hour — if holder hasn't released, assume dead


class ConsolidationLock:
    """Prevents two processes from consolidating simultaneously.

    Uses a lock file with PID and mtime in personality/ dir.
    """

    LOCK_FILE = ".consolidate-lock"
    LAST_CONSOLIDATED_FILE = ".last-consolidated-at"

    def __init__(self, personality_dir: Path) -> None:
        self.lock_path = personality_dir / self.LOCK_FILE
        self.last_consolidated_path = personality_dir / self.LAST_CONSOLIDATED_FILE

    async def try_acquire(self) -> bool:
        """Try to acquire lock. Returns False if held by another live process.

        Write our PID to lock file. Re-read to verify we won the race.
        """
        # Check for existing lock
        if self.lock_path.exists():
            try:
                existing_pid = int(self.lock_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                existing_pid = -1

            # Check if process is still alive
            if existing_pid > 0 and self._is_process_alive(existing_pid):
                # Check staleness
                mtime = self.lock_path.stat().st_mtime
                if time.time() - mtime < _STALE_SECONDS:
                    logger.info(
                        "consolidation_lock_held",
                        pid=existing_pid,
                    )
                    return False
                # Stale lock — reclaim
                logger.warning(
                    "consolidation_lock_stale_reclaim",
                    pid=existing_pid,
                )

        # Write our PID
        my_pid = os.getpid()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(str(my_pid), encoding="utf-8")

        # Re-read to verify we won the race
        try:
            written_pid = int(self.lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return False

        if written_pid != my_pid:
            logger.info("consolidation_lock_race_lost", winner=written_pid)
            return False

        logger.info("consolidation_lock_acquired", pid=my_pid)
        return True

    async def release(self) -> None:
        """Release lock by deleting lock file."""
        try:
            self.lock_path.unlink(missing_ok=True)
            logger.info("consolidation_lock_released")
        except OSError:
            logger.warning("consolidation_lock_release_failed")

    async def read_last_consolidated_at(self) -> float:
        """Return timestamp of the last successful consolidation.

        Returns 0.0 when no successful run has been recorded yet.
        """
        if not self.last_consolidated_path.exists():
            return 0.0
        try:
            return float(
                self.last_consolidated_path.read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            return 0.0

    async def mark_consolidated_at(self, timestamp: float | None = None) -> None:
        """Record a successful consolidation timestamp."""
        self.last_consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.time() if timestamp is None else timestamp
        self.last_consolidated_path.write_text(f"{ts:.6f}", encoding="utf-8")

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if a process with given PID exists."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # Process exists but we can't signal it
        return True
