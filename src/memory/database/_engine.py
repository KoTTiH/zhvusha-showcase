"""Engine and session-maker factories — internal, forbidden externally.

Imports SQLAlchemy primitives only. Exposed to external clients via
:mod:`src.memory.database` package re-exports.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_engine(database_url: str) -> AsyncEngine:
    """Create async SQLAlchemy engine."""
    return create_async_engine(database_url, echo=False)


def get_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)
