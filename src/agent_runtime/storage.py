"""Agent job storage implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.models import AgentJob, AgentJobStatus

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class AgentJobNotFoundError(KeyError):
    """Raised when a requested job id is absent."""


class AgentJobStore(Protocol):
    """Durable storage interface for runtime jobs."""

    async def create(self, job: AgentJob) -> AgentJob: ...
    async def save(self, job: AgentJob) -> AgentJob: ...
    async def get(self, job_id: str) -> AgentJob: ...
    async def find_by_fingerprint(self, fingerprint: str) -> AgentJob | None: ...
    async def list_by_status(
        self, statuses: tuple[AgentJobStatus, ...]
    ) -> list[AgentJob]: ...


class InMemoryAgentJobStore:
    """In-memory job store for contract tests and local adapters."""

    def __init__(self) -> None:
        self._jobs: dict[str, AgentJob] = {}

    async def create(self, job: AgentJob) -> AgentJob:
        self._jobs[job.id] = job
        return job

    async def save(self, job: AgentJob) -> AgentJob:
        self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> AgentJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise AgentJobNotFoundError(job_id) from exc

    async def find_by_fingerprint(self, fingerprint: str) -> AgentJob | None:
        return _preferred_fingerprint_match(
            job for job in self._jobs.values() if job.fingerprint == fingerprint
        )

    async def list_by_status(
        self, statuses: tuple[AgentJobStatus, ...]
    ) -> list[AgentJob]:
        wanted = set(statuses)
        return [job for job in self._jobs.values() if job.status in wanted]


class FileAgentJobStore:
    """Simple durable JSON-file store for AgentJob records."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def create(self, job: AgentJob) -> AgentJob:
        return await self.save(job)

    async def save(self, job: AgentJob) -> AgentJob:
        path = self._path_for(job.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        return job

    async def get(self, job_id: str) -> AgentJob:
        path = self._path_for(job_id)
        if not path.exists():
            raise AgentJobNotFoundError(job_id)
        return AgentJob.model_validate_json(path.read_text(encoding="utf-8"))

    async def find_by_fingerprint(self, fingerprint: str) -> AgentJob | None:
        matches: list[AgentJob] = []
        for path in self._job_paths():
            job = AgentJob.model_validate_json(path.read_text(encoding="utf-8"))
            if job.fingerprint == fingerprint:
                matches.append(job)
        return _preferred_fingerprint_match(matches)

    async def list_by_status(
        self, statuses: tuple[AgentJobStatus, ...]
    ) -> list[AgentJob]:
        wanted = set(statuses)
        jobs: list[AgentJob] = []
        for path in self._job_paths():
            job = AgentJob.model_validate_json(path.read_text(encoding="utf-8"))
            if job.status in wanted:
                jobs.append(job)
        return jobs

    def _path_for(self, job_id: str) -> Path:
        return self._root / "jobs" / f"{job_id}.json"

    def _job_paths(self) -> list[Path]:
        job_dir = self._root / "jobs"
        if not job_dir.exists():
            return []
        return sorted(job_dir.glob("*.json"))


_TERMINAL_STATUSES = {
    AgentJobStatus.DONE,
    AgentJobStatus.FAILED,
    AgentJobStatus.CANCELED,
    AgentJobStatus.NEEDS_REVIEW,
}


def _preferred_fingerprint_match(jobs: Iterable[AgentJob]) -> AgentJob | None:
    matches = list(jobs)
    if not matches:
        return None
    non_terminal = [job for job in matches if job.status not in _TERMINAL_STATUSES]
    candidates = non_terminal or matches
    return max(candidates, key=lambda job: (job.updated_at, job.created_at))
