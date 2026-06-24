"""Approval contracts for Agent Runtime side-effect capabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path

    from src.agent_runtime.models import AgentJob


@dataclass(frozen=True)
class AgentToolApproval:
    """A durable approval grant for high-risk Agent Runtime capabilities."""

    approval_id: str
    capabilities: tuple[str, ...]
    approved_by: int
    status: str = "approved"
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    @classmethod
    def approved(
        cls,
        *,
        approval_id: str,
        capabilities: tuple[str, ...],
        approved_by: int,
    ) -> AgentToolApproval:
        """Create an approved capability grant."""
        return cls(
            approval_id=approval_id,
            capabilities=tuple(sorted(set(capabilities))),
            approved_by=approved_by,
        )

    def allows(self, capability: str) -> bool:
        """Return whether this approval grants a capability."""
        return self.status == "approved" and capability in set(self.capabilities)


@dataclass(frozen=True)
class AgentToolApprovalGrant:
    """A durable scoped ToolGateway grant bound to one Agent Runtime job."""

    approval_id: str
    capabilities: tuple[str, ...]
    approved_by: int
    job_id: str
    owner_user_id: int
    chat_id: int
    source_message_id: str
    status: str = "approved"
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    metadata: dict[str, str] = field(default_factory=dict)

    def to_approval(self) -> AgentToolApproval:
        """Return the ToolGateway approval value for this grant."""
        return AgentToolApproval(
            approval_id=self.approval_id,
            capabilities=tuple(sorted(set(self.capabilities))),
            approved_by=self.approved_by,
            status=self.status,
            created_at=self.created_at,
        )

    def matches_job(self, job: AgentJob) -> bool:
        """Return whether the grant is scoped to this exact runtime job."""
        return (
            self.status == "approved"
            and self.job_id == job.id
            and self.owner_user_id == job.owner_user_id
            and self.chat_id == job.chat_id
            and self.source_message_id == job.source_message_id
        )

    def grants_all(self, capabilities: tuple[str, ...]) -> bool:
        """Return whether every required capability is granted."""
        return set(capabilities).issubset(set(self.capabilities))


class AgentToolApprovalGrantStore(Protocol):
    """Storage contract for durable scoped ToolGateway grants."""

    def issue_grant(
        self,
        *,
        approval_id: str,
        capabilities: tuple[str, ...],
        approved_by: int,
        job_id: str,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        metadata: dict[str, str] | None = None,
    ) -> AgentToolApprovalGrant: ...

    def get(self, approval_id: str) -> AgentToolApprovalGrant: ...

    def approved_for_job(
        self,
        *,
        job: AgentJob,
        approval_id: str,
        capabilities: tuple[str, ...],
    ) -> AgentToolApproval | None: ...


class InMemoryAgentToolApprovalGrantStore:
    """Process-local grant store for contract tests."""

    def __init__(self) -> None:
        self._grants: dict[str, AgentToolApprovalGrant] = {}

    def issue_grant(
        self,
        *,
        approval_id: str,
        capabilities: tuple[str, ...],
        approved_by: int,
        job_id: str,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        metadata: dict[str, str] | None = None,
    ) -> AgentToolApprovalGrant:
        grant = AgentToolApprovalGrant(
            approval_id=approval_id or f"tool-grant-{uuid4().hex}",
            capabilities=tuple(sorted(set(capabilities))),
            approved_by=approved_by,
            job_id=job_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            metadata=dict(metadata or {}),
        )
        self._grants[grant.approval_id] = grant
        return grant

    def get(self, approval_id: str) -> AgentToolApprovalGrant:
        return self._grants[approval_id]

    def approved_for_job(
        self,
        *,
        job: AgentJob,
        approval_id: str,
        capabilities: tuple[str, ...],
    ) -> AgentToolApproval | None:
        grant = self._grants.get(approval_id)
        if (
            grant is None
            or not grant.matches_job(job)
            or not grant.grants_all(capabilities)
        ):
            return None
        return grant.to_approval()


class FileAgentToolApprovalGrantStore:
    """Small durable file store for scoped ToolGateway grants."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def issue_grant(
        self,
        *,
        approval_id: str,
        capabilities: tuple[str, ...],
        approved_by: int,
        job_id: str,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        metadata: dict[str, str] | None = None,
    ) -> AgentToolApprovalGrant:
        grant = AgentToolApprovalGrant(
            approval_id=approval_id or f"tool-grant-{uuid4().hex}",
            capabilities=tuple(sorted(set(capabilities))),
            approved_by=approved_by,
            job_id=job_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            metadata=dict(metadata or {}),
        )
        self._write(grant)
        return grant

    def get(self, approval_id: str) -> AgentToolApprovalGrant:
        data = json.loads(self._path(approval_id).read_text(encoding="utf-8"))
        capabilities = data.get("capabilities", ())
        if isinstance(capabilities, list):
            data["capabilities"] = tuple(capabilities)
        return AgentToolApprovalGrant(**data)

    def approved_for_job(
        self,
        *,
        job: AgentJob,
        approval_id: str,
        capabilities: tuple[str, ...],
    ) -> AgentToolApproval | None:
        try:
            grant = self.get(approval_id)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not grant.matches_job(job) or not grant.grants_all(capabilities):
            return None
        return grant.to_approval()

    def _write(self, grant: AgentToolApprovalGrant) -> None:
        path = self._path(grant.approval_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(asdict(grant), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def _path(self, approval_id: str) -> Path:
        return self._root / f"{approval_id}.json"


@dataclass(frozen=True)
class AgentApprovalRequest:
    """Audit record for a requested side-effect approval."""

    approval_id: str
    job_id: str
    capability: str
    reason: str
    requested_by: int
    status: str = "pending"
    telegram_status: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    resolved_at: str = ""
    approved_by: int | None = None


class FileAgentApprovalStore:
    """Small durable file store for Agent Runtime approval requests."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def create_request(
        self,
        *,
        job_id: str,
        capability: str,
        reason: str,
        requested_by: int,
        telegram_status: str = "",
    ) -> AgentApprovalRequest:
        """Persist a pending approval request."""
        request = AgentApprovalRequest(
            approval_id=f"approval-{uuid4().hex}",
            job_id=job_id,
            capability=capability,
            reason=reason,
            requested_by=requested_by,
            telegram_status=telegram_status,
        )
        self._write(request)
        return request

    def approve(self, approval_id: str, *, approved_by: int) -> AgentToolApproval:
        """Mark a request approved and return the granted capability."""
        request = self.get(approval_id)
        resolved = AgentApprovalRequest(
            approval_id=request.approval_id,
            job_id=request.job_id,
            capability=request.capability,
            reason=request.reason,
            requested_by=request.requested_by,
            status="approved",
            telegram_status=request.telegram_status,
            created_at=request.created_at,
            resolved_at=datetime.now(UTC).isoformat(),
            approved_by=approved_by,
        )
        self._write(resolved)
        return AgentToolApproval.approved(
            approval_id=approval_id,
            capabilities=(request.capability,),
            approved_by=approved_by,
        )

    def get(self, approval_id: str) -> AgentApprovalRequest:
        """Load a persisted approval request."""
        path = self._path(approval_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentApprovalRequest(**data)

    def _write(self, request: AgentApprovalRequest) -> None:
        self._path(request.approval_id).write_text(
            json.dumps(asdict(request), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _path(self, approval_id: str) -> Path:
        return self._root / f"{approval_id}.json"
