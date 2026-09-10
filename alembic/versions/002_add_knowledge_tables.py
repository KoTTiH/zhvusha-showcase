"""Add knowledge base tables.

Revision ID: 002
Create Date: 2026-04-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: str = "001"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Enable ltree extension for hierarchical categories
    op.execute("CREATE EXTENSION IF NOT EXISTS ltree;")

    # --- Categories tree ---
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_ru", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False, unique=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("entry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("categories.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.execute(
        "CREATE INDEX idx_cat_path ON categories USING GIST (CAST(path AS ltree));"
    )

    # --- Knowledge entries ---
    op.create_table(
        "knowledge_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "category_id",
            sa.Integer(),
            sa.ForeignKey("categories.id"),
            nullable=True,
        ),
        sa.Column(
            "tags",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "content_type",
            sa.Text(),
            nullable=False,
            server_default="fact",
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="raw",
        ),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )

    # HNSW vector index for semantic search
    op.execute(
        "CREATE INDEX idx_ke_embedding ON knowledge_entries "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m=16, ef_construction=200);"
    )
    # Full-text search (Russian config)
    op.execute(
        "CREATE INDEX idx_ke_fts ON knowledge_entries "
        "USING gin (to_tsvector('russian', "
        "coalesce(title,'') || ' ' || coalesce(content,'')));"
    )
    # Tags GIN index
    op.execute("CREATE INDEX idx_ke_tags ON knowledge_entries USING gin (tags);")
    op.create_index("idx_ke_category", "knowledge_entries", ["category_id"])
    op.create_index("idx_ke_status", "knowledge_entries", ["status"])

    # --- Entry relations ---
    op.create_table(
        "entry_relations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation_type", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- Knowledge staging (Sleep-Time Agent proposals) ---
    op.create_table(
        "knowledge_staging",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column(
            "target_entry_id",
            sa.Integer(),
            sa.ForeignKey("knowledge_entries.id"),
            nullable=True,
        ),
        sa.Column("proposed_changes", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("proposed_by", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("knowledge_staging")
    op.drop_table("entry_relations")
    op.drop_table("knowledge_entries")
    op.drop_table("categories")
