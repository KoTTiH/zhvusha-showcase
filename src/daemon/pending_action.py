"""Pending action model and approval store for daemon approval flow.

Daemon stores pending actions here when safety guard requires approval.
Bot middleware updates status when user replies. Daemon polls for approved
actions and executes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import BigInteger, CursorResult, DateTime, Text, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.memory.database import Base

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = structlog.get_logger()


class ActionStatus(StrEnum):
    """Valid statuses for pending actions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class PendingActionDTO:
    """Detached projection of PendingAction — safe to use outside session scope."""

    id: int
    signal_id: str
    tool_name: str
    tool_params: dict[str, Any] | None
    decision_type: str
    reasoning: str | None
    safety_reason: str | None
    status: str
    telegram_message_id: int | None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    claimed_at: datetime | None = None
    executed_at: datetime | None = None


class PendingAction(Base):
    """A daemon action awaiting user approval."""

    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    tool_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    decision_type: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    safety_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    telegram_message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def to_dto(self) -> PendingActionDTO:
        """Project into a detached DTO safe for use outside session."""
        return PendingActionDTO(
            id=self.id,
            signal_id=self.signal_id,
            tool_name=self.tool_name,
            tool_params=self.tool_params,
            decision_type=self.decision_type,
            reasoning=self.reasoning,
            safety_reason=self.safety_reason,
            status=self.status,
            telegram_message_id=self.telegram_message_id,
            created_at=self.created_at,
            resolved_at=self.resolved_at,
            claimed_at=self.claimed_at,
            executed_at=self.executed_at,
        )


class ApprovalStore:
    """Data access for pending approval actions."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_maker = session_maker

    async def create(
        self,
        *,
        signal_id: str,
        tool_name: str,
        tool_params: dict[str, Any],
        decision_type: str,
        reasoning: str,
        safety_reason: str,
    ) -> int:
        """Create a pending action. Returns its ID."""
        entry = PendingAction(
            signal_id=signal_id,
            tool_name=tool_name,
            tool_params=tool_params,
            decision_type=decision_type,
            reasoning=reasoning,
            safety_reason=safety_reason,
        )
        async with self._session_maker() as session:
            session.add(entry)
            await session.flush()
            action_id: int = entry.id
            await session.commit()

        logger.info("pending_action_created", id=action_id, tool=tool_name)
        return action_id

    async def set_status(self, action_id: int, status: ActionStatus) -> bool:
        """Update status (pending → approved/rejected). Atomic via WHERE.

        Returns True if updated, False if not found or already resolved.
        """
        async with self._session_maker() as session:
            stmt = (
                update(PendingAction)
                .where(
                    PendingAction.id == action_id,
                    PendingAction.status == ActionStatus.PENDING,
                )
                .values(status=status, resolved_at=datetime.now(tz=UTC))
            )
            result: CursorResult[Any] = await session.execute(stmt)  # type: ignore[assignment]
            await session.commit()
            updated: bool = result.rowcount > 0

        if updated:
            logger.info("pending_action_status", id=action_id, status=status)
        return updated

    async def set_telegram_message_id(self, action_id: int, message_id: int) -> None:
        """Store the Telegram message ID for reply matching."""
        async with self._session_maker() as session:
            stmt = (
                update(PendingAction)
                .where(PendingAction.id == action_id)
                .values(telegram_message_id=message_id)
            )
            await session.execute(stmt)
            await session.commit()

    async def get_by_id(self, action_id: int) -> PendingActionDTO | None:
        """Find a pending action by primary key."""
        async with self._session_maker() as session:
            stmt = select(PendingAction).where(PendingAction.id == action_id)
            result = await session.execute(stmt)
            row = result.scalars().first()
            return row.to_dto() if row else None

    async def get_by_telegram_message_id(
        self, message_id: int
    ) -> PendingActionDTO | None:
        """Find a pending action by its Telegram notification message ID."""
        async with self._session_maker() as session:
            stmt = select(PendingAction).where(
                PendingAction.telegram_message_id == message_id
            )
            result = await session.execute(stmt)
            row = result.scalars().first()
            return row.to_dto() if row else None

    async def get_approved(self, limit: int = 10) -> list[PendingActionDTO]:
        """Fetch actions with status='approved', oldest first."""
        async with self._session_maker() as session:
            stmt = (
                select(PendingAction)
                .where(PendingAction.status == ActionStatus.APPROVED)
                .order_by(PendingAction.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [row.to_dto() for row in result.scalars().all()]

    async def mark_executing(self, action_id: int) -> bool:
        """Atomically claim action for execution (approved → executing).

        Returns True if claimed, False if already claimed or not approved.
        """
        async with self._session_maker() as session:
            stmt = (
                update(PendingAction)
                .where(
                    PendingAction.id == action_id,
                    PendingAction.status == ActionStatus.APPROVED,
                )
                .values(status=ActionStatus.EXECUTING, claimed_at=datetime.now(tz=UTC))
            )
            result: CursorResult[Any] = await session.execute(stmt)  # type: ignore[assignment]
            await session.commit()
            claimed: bool = result.rowcount > 0

        if claimed:
            logger.info("pending_action_claimed", id=action_id)
        return claimed

    async def mark_executed(self, action_id: int, *, success: bool) -> bool:
        """Mark action as executed or failed.

        Returns True if updated, False if action was no longer in 'executing'
        state (e.g. recovered by recover_stuck).
        """
        status = ActionStatus.EXECUTED if success else ActionStatus.FAILED
        async with self._session_maker() as session:
            stmt = (
                update(PendingAction)
                .where(
                    PendingAction.id == action_id,
                    PendingAction.status == ActionStatus.EXECUTING,
                )
                .values(status=status, executed_at=datetime.now(tz=UTC))
            )
            result: CursorResult[Any] = await session.execute(stmt)  # type: ignore[assignment]
            await session.commit()
            updated: bool = result.rowcount > 0

        if updated:
            logger.info("pending_action_executed", id=action_id, status=status)
        else:
            logger.warning("mark_executed_stale", id=action_id)
        return updated

    async def recover_stuck(self, timeout_minutes: int = 10) -> int:
        """Reset actions stuck in 'executing' for longer than timeout back to 'approved'.

        Returns the number of recovered actions.
        """
        cutoff = datetime.now(tz=UTC) - timedelta(minutes=timeout_minutes)
        async with self._session_maker() as session:
            stmt = (
                update(PendingAction)
                .where(
                    PendingAction.status == ActionStatus.EXECUTING,
                    PendingAction.claimed_at < cutoff,
                )
                .values(status=ActionStatus.APPROVED, claimed_at=None)
            )
            result: CursorResult[Any] = await session.execute(stmt)  # type: ignore[assignment]
            await session.commit()
            count: int = result.rowcount

        if count > 0:
            logger.warning(
                "stuck_actions_recovered", count=count, timeout_minutes=timeout_minutes
            )
        return count
