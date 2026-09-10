"""Convert audit_log JSON columns to JSONB to match the ORM.

Revision ID: 005
Create Date: 2026-04-20

Background:
    ``src.daemon.audit.AuditLog`` declares ``tool_params`` and ``result_details``
    as ``JSONB``, but migration ``003`` created the columns as ``JSON``.
    This ORM/schema drift was caught during the phase-8 validation sweep.

Rollback-safe: Postgres ``ALTER TYPE JSON <-> JSONB USING col::target`` is
a valid cast both directions. No data is lost (JSONB is a strict superset
of JSON).
"""

from __future__ import annotations

from alembic import op

revision: str = "005"
down_revision: str = "004"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE audit_log "
        "ALTER COLUMN tool_params TYPE JSONB USING tool_params::jsonb"
    )
    op.execute(
        "ALTER TABLE audit_log "
        "ALTER COLUMN result_details TYPE JSONB USING result_details::jsonb"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE audit_log "
        "ALTER COLUMN tool_params TYPE JSON USING tool_params::json"
    )
    op.execute(
        "ALTER TABLE audit_log "
        "ALTER COLUMN result_details TYPE JSON USING result_details::json"
    )
