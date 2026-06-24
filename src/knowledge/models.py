"""SQLAlchemy models for the knowledge base."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003  # SQLAlchemy needs runtime access

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.memory.database import Base


class Category(Base):
    """Hierarchical category for knowledge entries (ltree paths)."""

    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    name_ru: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text, unique=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    entries: Mapped[list[KnowledgeEntry]] = relationship(
        back_populates="category", lazy="selectin"
    )


class KnowledgeEntry(Base):
    """Single knowledge base entry with embedding and metadata."""

    __tablename__ = "knowledge_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("categories.id"), nullable=True
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str] = mapped_column(Text, server_default="fact")
    status: Mapped[str] = mapped_column(Text, server_default="raw")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata", JSONB, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    category: Mapped[Category | None] = relationship(
        back_populates="entries", lazy="joined"
    )


class EntryRelation(Base):
    """Directed relation between two knowledge entries."""

    __tablename__ = "entry_relations"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_entries.id", ondelete="CASCADE")
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_entries.id", ondelete="CASCADE")
    )
    relation_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KnowledgeStagingItem(Base):
    """Proposed change from Sleep-Time Agent, pending review."""

    __tablename__ = "knowledge_staging"

    id: Mapped[int] = mapped_column(primary_key=True)
    operation: Mapped[str] = mapped_column(Text)
    target_entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_entries.id"), nullable=True
    )
    proposed_changes: Mapped[dict | None] = mapped_column(  # type: ignore[type-arg]
        JSONB, nullable=True
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    proposed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
