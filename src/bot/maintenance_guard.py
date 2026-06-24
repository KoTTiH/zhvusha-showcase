"""File-backed maintenance guards shared by bot startup loops."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = structlog.get_logger()

_DEFAULT_MORNING_MARKER_TTL_SECONDS = 6 * 60 * 60


class MaintenanceGuardError(RuntimeError):
    """Raised when a maintenance guard cannot fail safely."""


@dataclass(frozen=True)
class AutonomousSelfCodingGuardDecision:
    should_run: bool
    reason: str
    last_run_at: float | None = None
    next_allowed_at: float | None = None
    marker_path: str | None = None
    idle_seconds: float | None = None


def default_runtime_dir(workspace_root: Path) -> Path:
    return workspace_root / "runtime"


def default_morning_maintenance_marker_path(workspace_root: Path) -> Path:
    return default_runtime_dir(workspace_root) / "maintenance" / "morning.json"


def default_autonomous_self_coding_state_path(workspace_root: Path) -> Path:
    return default_runtime_dir(workspace_root) / "autonomous_self_coding.json"


def default_user_activity_path(workspace_root: Path) -> Path:
    return default_runtime_dir(workspace_root) / "user_activity.json"


def resolve_autonomous_self_coding_state_path(
    raw_state_path: str,
    workspace_root: Path,
) -> Path:
    value = raw_state_path.strip()
    if not value:
        return default_autonomous_self_coding_state_path(workspace_root)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return workspace_root / path


def resolve_user_activity_path(raw_activity_path: str, workspace_root: Path) -> Path:
    value = raw_activity_path.strip()
    if not value:
        return default_user_activity_path(workspace_root)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return workspace_root / path


class MorningMaintenanceGuard:
    """A short-lived file marker for active /morning maintenance windows."""

    def __init__(
        self,
        *,
        marker_path: Path,
        ttl_seconds: int = _DEFAULT_MORNING_MARKER_TTL_SECONDS,
    ) -> None:
        self.marker_path = marker_path
        self.ttl_seconds = max(1, ttl_seconds)

    @contextmanager
    def active(
        self,
        *,
        owner: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        token = f"{os.getpid()}-{time.time_ns()}"
        if not self.mark_active(owner=owner, token=token, metadata=metadata):
            raise MaintenanceGuardError(
                f"Could not persist maintenance marker at {self.marker_path}"
            )
        try:
            yield
        finally:
            self.release(token=token)

    def mark_active(
        self,
        *,
        owner: str,
        token: str,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> bool:
        started_at = time.time() if now is None else now
        payload = {
            "owner": owner,
            "pid": os.getpid(),
            "started_at": started_at,
            "expires_at": started_at + self.ttl_seconds,
            "token": token,
            "metadata": metadata or {},
        }
        return _write_json(self._token_marker_path(token), payload)

    def release(self, *, token: str | None = None) -> None:
        release_path = self._token_marker_path(token) if token else self.marker_path
        try:
            release_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "morning_maintenance_marker_release_failed",
                marker_path=str(release_path),
                exc_info=True,
            )

    def is_active(self, *, now: float | None = None) -> bool:
        checked_at = time.time() if now is None else now
        return any(
            self._marker_path_is_active(path, now=checked_at)
            for path in self._marker_paths()
        )

    def _marker_path_is_active(self, path: Path, *, now: float) -> bool:
        started_at = self._marker_started_at(path)
        if started_at is None:
            return False
        if now - started_at <= self.ttl_seconds:
            return True
        logger.warning(
            "morning_maintenance_marker_stale_ignored",
            marker_path=str(path),
            age_seconds=now - started_at,
            ttl_seconds=self.ttl_seconds,
        )
        return False

    def _marker_started_at(self, path: Path) -> float | None:
        payload = _read_json(path)
        started_at = _coerce_float(payload.get("started_at")) if payload else None
        if started_at is not None:
            return started_at
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _marker_paths(self) -> tuple[Path, ...]:
        paths = [self.marker_path]
        try:
            paths.extend(
                sorted(self.marker_path.parent.glob(f"{self.marker_path.stem}-*.json"))
            )
        except OSError:
            logger.warning(
                "morning_maintenance_marker_scan_failed",
                marker_path=str(self.marker_path),
                exc_info=True,
            )
        return tuple(path for path in paths if path.exists())

    def _token_marker_path(self, token: str | None) -> Path:
        if not token:
            return self.marker_path
        return self.marker_path.with_name(f"{self.marker_path.stem}-{token}.json")


class UserActivityGuard:
    """Persistent marker for the last direct user activity seen by the bot."""

    def __init__(
        self,
        *,
        activity_path: Path,
        idle_seconds: int,
    ) -> None:
        self.activity_path = activity_path
        self.idle_seconds = max(0, idle_seconds)

    def record_activity(
        self,
        *,
        source: str,
        now: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        recorded_at = time.time() if now is None else now
        payload = {
            "last_user_activity_at": recorded_at,
            "source": source,
            "pid": os.getpid(),
            "updated_at": recorded_at,
            "metadata": metadata or {},
        }
        return _write_json(self.activity_path, payload)

    def should_allow_autonomous_run(
        self,
        *,
        now: float | None = None,
    ) -> AutonomousSelfCodingGuardDecision:
        checked_at = time.time() if now is None else now
        last_activity_at, state_ok = self._load_last_activity_at()
        if not state_ok or last_activity_at is None:
            return AutonomousSelfCodingGuardDecision(
                should_run=False,
                reason="user_activity_unknown",
                marker_path=str(self.activity_path),
            )
        next_allowed_at = last_activity_at + self.idle_seconds
        if checked_at < next_allowed_at:
            return AutonomousSelfCodingGuardDecision(
                should_run=False,
                reason="user_recently_active",
                last_run_at=last_activity_at,
                next_allowed_at=next_allowed_at,
                marker_path=str(self.activity_path),
                idle_seconds=max(0.0, checked_at - last_activity_at),
            )
        return AutonomousSelfCodingGuardDecision(
            should_run=True,
            reason="user_idle",
            last_run_at=last_activity_at,
            marker_path=str(self.activity_path),
            idle_seconds=max(0.0, checked_at - last_activity_at),
        )

    def _load_last_activity_at(self) -> tuple[float | None, bool]:
        try:
            state_exists = self.activity_path.exists()
        except OSError:
            logger.warning(
                "user_activity_state_exists_check_failed",
                activity_path=str(self.activity_path),
                exc_info=True,
            )
            return None, False
        if not state_exists:
            return None, True
        payload = _read_json(self.activity_path)
        if payload is None:
            return None, False
        last_activity_at = _coerce_float(payload.get("last_user_activity_at"))
        if last_activity_at is None:
            logger.warning(
                "user_activity_state_missing_last_user_activity_at",
                activity_path=str(self.activity_path),
            )
            return None, False
        return last_activity_at, True


class AutonomousSelfCodingRuntimeGuard:
    """Persistent throttle and maintenance-window guard for scheduled self-coding."""

    def __init__(
        self,
        *,
        state_path: Path,
        restart_throttle_seconds: int,
        morning_guard: MorningMaintenanceGuard | None = None,
        user_activity_guard: UserActivityGuard | None = None,
    ) -> None:
        self.state_path = state_path
        self.restart_throttle_seconds = max(0, restart_throttle_seconds)
        self.morning_guard = morning_guard
        self.user_activity_guard = user_activity_guard

    def should_run(
        self,
        *,
        now: float | None = None,
    ) -> AutonomousSelfCodingGuardDecision:
        checked_at = time.time() if now is None else now
        if self.morning_guard is not None and self.morning_guard.is_active(
            now=checked_at
        ):
            return AutonomousSelfCodingGuardDecision(
                should_run=False,
                reason="morning_maintenance_active",
                marker_path=str(self.morning_guard.marker_path),
            )

        if self.user_activity_guard is not None:
            user_activity = self.user_activity_guard.should_allow_autonomous_run(
                now=checked_at,
            )
            if not user_activity.should_run:
                return user_activity

        last_run_at, state_ok = self._load_last_run_at()
        if not state_ok:
            return AutonomousSelfCodingGuardDecision(
                should_run=False,
                reason="persistent_state_unreadable",
                marker_path=str(self.state_path),
            )
        if last_run_at is None or self.restart_throttle_seconds <= 0:
            return AutonomousSelfCodingGuardDecision(should_run=True, reason="allowed")

        next_allowed_at = last_run_at + self.restart_throttle_seconds
        if checked_at < next_allowed_at:
            return AutonomousSelfCodingGuardDecision(
                should_run=False,
                reason="restart_throttle_active",
                last_run_at=last_run_at,
                next_allowed_at=next_allowed_at,
            )
        return AutonomousSelfCodingGuardDecision(
            should_run=True,
            reason="allowed",
            last_run_at=last_run_at,
        )

    def last_run_at(self) -> float | None:
        last_run_at, state_ok = self._load_last_run_at()
        return last_run_at if state_ok else None

    def _load_last_run_at(self) -> tuple[float | None, bool]:
        try:
            state_exists = self.state_path.exists()
        except OSError:
            logger.warning(
                "autonomous_self_coding_state_exists_check_failed",
                state_path=str(self.state_path),
                exc_info=True,
            )
            return None, False
        if not state_exists:
            return None, True
        payload = _read_json(self.state_path)
        if payload is None:
            return None, False
        last_run_at = _coerce_float(payload.get("last_run_at"))
        if last_run_at is None:
            logger.warning(
                "autonomous_self_coding_state_missing_last_run_at",
                state_path=str(self.state_path),
            )
            return None, False
        return last_run_at, True

    def record_run(
        self,
        *,
        status: str,
        now: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        recorded_at = time.time() if now is None else now
        payload = {
            "last_run_at": recorded_at,
            "status": status,
            "pid": os.getpid(),
            "updated_at": recorded_at,
            "metadata": metadata or {},
        }
        return _write_json(self.state_path, payload)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("maintenance_guard_json_invalid", path=str(path))
        return None
    if not isinstance(data, dict):
        logger.warning("maintenance_guard_json_not_object", path=str(path))
        return None
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return True
    except OSError:
        logger.warning(
            "maintenance_guard_write_failed",
            path=str(path),
            exc_info=True,
        )
        return False
