"""Add episodes table.

Revision ID: 001
Create Date: 2026-04-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "episodes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Who
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("chat_type", sa.String(20), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        # Content
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        # Vector embedding (384-dim, paraphrase-multilingual-MiniLM-L12-v2)
        sa.Column("embedding", Vector(384), nullable=True),
        # Importance and somatic marker
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("valence", sa.String(20), nullable=False, server_default="neutral"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        # Consolidation
        sa.Column(
            "consolidated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("consolidation_result", sa.Text(), nullable=True),
        # ACT-R retrieval tracking
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        # Enrichment
        sa.Column(
            "enrichment_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("intent", sa.String(30), nullable=True),
        sa.Column("emotion", sa.String(30), nullable=True),
        sa.Column(
            "embedding_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        # Metadata
        sa.Column("source", sa.String(50), nullable=False, server_default="chat"),
        sa.Column("metadata_json", sa.Text(), nullable=True),
    )

    # HNSW vector index (no training step, good for INSERT-heavy)
    op.execute(
        "CREATE INDEX ix_episodes_embedding ON episodes "
        "USING hnsw (embedding vector_cosine_ops);"
    )

    # B-tree indexes
    op.create_index("ix_episodes_timestamp", "episodes", ["timestamp"])
    op.create_index("ix_episodes_user_id", "episodes", ["user_id"])
    op.create_index("ix_episodes_consolidated", "episodes", ["consolidated"])
    op.create_index("ix_episodes_source", "episodes", ["source"])
    op.create_index("ix_episodes_intent", "episodes", ["intent"])
    op.create_index("ix_episodes_emotion", "episodes", ["emotion"])
    op.create_index("ix_episodes_enrichment_status", "episodes", ["enrichment_status"])


def downgrade() -> None:
    op.drop_table("episodes")
