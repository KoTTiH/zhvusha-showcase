"""Add self-coding archive nodes.

Revision ID: 007
Create Date: 2026-05-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "007"
down_revision: str = "006"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "archive_nodes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slug", sa.Text(), nullable=False, unique=True),
        sa.Column("spec_slug", sa.Text(), nullable=True),
        sa.Column("proposal_slug", sa.Text(), nullable=True),
        sa.Column("tier", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("branch", sa.Text(), nullable=True),
        sa.Column("commit_sha", sa.Text(), nullable=True),
        sa.Column("parent_slug", sa.Text(), nullable=True),
        sa.Column("diff_summary", sa.Text(), nullable=False),
        sa.Column("tests_summary", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False, server_default=""),
        sa.Column("insight", sa.Text(), nullable=False),
        sa.Column(
            "source_evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "model_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_archive_nodes_status", "archive_nodes", ["status"])
    op.create_index("idx_archive_nodes_spec", "archive_nodes", ["spec_slug"])
    op.create_index("idx_archive_nodes_commit", "archive_nodes", ["commit_sha"])
    op.create_index("idx_archive_nodes_created", "archive_nodes", ["created_at"])


def downgrade() -> None:
    op.drop_table("archive_nodes")
