"""Alembic environment for async SQLAlchemy migrations."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

import src.daemon.audit  # register AuditLogEntry with Base.metadata
import src.daemon.pending_action  # register PendingAction
import src.knowledge.models  # noqa: F401  — register knowledge models
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from src.memory.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Get database URL from environment or alembic.ini."""
    return os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", ""),
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: object) -> None:  # type: ignore[type-arg]
    """Run migrations with a connection."""
    context.configure(connection=connection, target_metadata=target_metadata)  # type: ignore[arg-type]

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = create_async_engine(get_url(), echo=False)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
