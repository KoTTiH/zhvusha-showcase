"""File-backed live process ownership guard."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class ProcessOwnerLease(BaseModel):
    """One live service owner lease."""

    model_config = ConfigDict(extra="ignore")

    service: str
    owner_id: str
    pid: int = Field(ge=0)
    acquired_at: str = Field(default_factory=_now_iso)
    heartbeat_at: str = Field(default_factory=_now_iso)

    @field_validator("service", "owner_id", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()


class ProcessOwnershipStatus(BaseModel):
    """Acquire/status result for process ownership."""

    service: str
    acquired: bool = False
    reason: str = ""
    owner: ProcessOwnerLease | None = None


class FileProcessOwnershipGuard:
    """Persist owner leases and reject duplicate live owners."""

    def __init__(
        self,
        path: Path,
        *,
        pid_is_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self._path = path
        self._pid_is_alive = pid_is_alive or _pid_is_alive

    def acquire(
        self,
        *,
        service: str,
        owner_id: str,
        pid: int | None = None,
    ) -> ProcessOwnershipStatus:
        service_id = _clean_service(service)
        owner = ProcessOwnerLease(
            service=service_id,
            owner_id=owner_id,
            pid=pid if pid is not None else os.getpid(),
        )
        with _exclusive_file_lock(self._path):
            leases = self._load()
            existing = leases.get(service_id)
            if existing is not None and self._pid_is_alive(existing.pid):
                return ProcessOwnershipStatus(
                    service=service_id,
                    acquired=False,
                    reason="already_owned",
                    owner=existing,
                )
            leases[service_id] = owner
            self._save(leases)
        return ProcessOwnershipStatus(
            service=service_id,
            acquired=True,
            reason="stale_replaced" if existing is not None else "acquired",
            owner=owner,
        )

    def status(self, service: str) -> ProcessOwnershipStatus:
        service_id = _clean_service(service)
        with _exclusive_file_lock(self._path):
            owner = self._load().get(service_id)
        if owner is None:
            return ProcessOwnershipStatus(service=service_id, reason="not_owned")
        alive = self._pid_is_alive(owner.pid)
        return ProcessOwnershipStatus(
            service=service_id,
            acquired=alive,
            reason="owned" if alive else "stale",
            owner=owner,
        )

    def release(self, service: str, *, owner_id: str) -> ProcessOwnershipStatus:
        service_id = _clean_service(service)
        with _exclusive_file_lock(self._path):
            leases = self._load()
            owner = leases.get(service_id)
            if owner is None:
                return ProcessOwnershipStatus(service=service_id, reason="not_owned")
            if owner.owner_id != owner_id:
                return ProcessOwnershipStatus(
                    service=service_id,
                    acquired=False,
                    reason="owner_mismatch",
                    owner=owner,
                )
            leases.pop(service_id, None)
            self._save(leases)
        return ProcessOwnershipStatus(
            service=service_id,
            acquired=True,
            reason="released",
            owner=owner,
        )

    def heartbeat(self, service: str, *, owner_id: str) -> ProcessOwnershipStatus:
        """Refresh heartbeat for the current owner without changing ownership."""
        service_id = _clean_service(service)
        with _exclusive_file_lock(self._path):
            leases = self._load()
            owner = leases.get(service_id)
            if owner is None:
                return ProcessOwnershipStatus(service=service_id, reason="not_owned")
            if owner.owner_id != owner_id:
                return ProcessOwnershipStatus(
                    service=service_id,
                    acquired=False,
                    reason="owner_mismatch",
                    owner=owner,
                )
            updated = owner.model_copy(update={"heartbeat_at": _now_iso()})
            leases[service_id] = updated
            self._save(leases)
        return ProcessOwnershipStatus(
            service=service_id,
            acquired=True,
            reason="heartbeat",
            owner=updated,
        )

    def _load(self) -> dict[str, ProcessOwnerLease]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8") or "{}")
        if not isinstance(raw, dict):
            return {}
        leases: dict[str, ProcessOwnerLease] = {}
        for service, payload in raw.items():
            if isinstance(payload, dict):
                leases[str(service)] = ProcessOwnerLease.model_validate(payload)
        return leases

    def _save(self, leases: dict[str, ProcessOwnerLease]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            service: lease.model_dump(mode="json")
            for service, lease in sorted(leases.items())
        }
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize lease file read/modify/write across live processes."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def render_process_ownership_report(
    statuses: tuple[ProcessOwnershipStatus, ...],
) -> str:
    """Render process ownership status for operator/chat diagnostics."""
    lines = ["Process ownership:"]
    if not statuses:
        lines.append("- no services checked.")
        return "\n".join(lines)
    for status in statuses:
        if status.owner is None:
            lines.append(f"- {status.service}: {status.reason}")
            continue
        owner = status.owner
        lines.append(
            "- "
            f"{status.service}: {status.reason}, "
            f"owner={owner.owner_id}, pid={owner.pid}, "
            f"heartbeat={owner.heartbeat_at}"
        )
    lines.append(
        "Если service уже owned живым PID, второй poller/startup должен остановиться "
        "до Telegram getUpdates conflict loop."
    )
    return "\n".join(lines)


def _clean_service(service: str) -> str:
    cleaned = str(service or "").strip()
    if not cleaned:
        raise ValueError("service must be non-empty")
    return cleaned
