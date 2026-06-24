"""Audit log for daemon actions — every decision is recorded."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import DateTime, Integer, Numeric, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.memory.database import Base

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = structlog.get_logger()


class AuditLogEntry(Base):
    """Single audit log record."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_params: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB, nullable=True
    )
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_details: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB, nullable=True
    )
    llm_tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AuditLog:
    """Data access for the audit log."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_maker = session_maker

    async def record(
        self,
        *,
        signal_id: str,
        decision: str,
        reasoning: str = "",
        tool_name: str | None = None,
        tool_params: dict[str, Any] | None = None,
        result: str = "",
        result_details: dict[str, Any] | None = None,
        llm_tokens_used: int = 0,
        llm_cost_usd: float = 0.0,
    ) -> int:
        """Record an audit entry. Returns entry ID."""
        entry = AuditLogEntry(
            signal_id=signal_id,
            decision=decision,
            reasoning=reasoning,
            tool_name=tool_name,
            tool_params=tool_params,
            result=result,
            result_details=result_details,
            llm_tokens_used=llm_tokens_used,
            llm_cost_usd=Decimal(str(llm_cost_usd)),
        )

        async with self._session_maker() as session:
            session.add(entry)
            await session.flush()
            entry_id: int = entry.id
            await session.commit()

        logger.info(
            "audit_recorded",
            id=entry_id,
            decision=decision,
            tool=tool_name,
            result=result,
        )
        return entry_id

    async def get_recent(self, limit: int = 50) -> list[AuditLogEntry]:
        """Get recent audit entries."""
        async with self._session_maker() as session:
            stmt = (
                select(AuditLogEntry)
                .order_by(AuditLogEntry.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
