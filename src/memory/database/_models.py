"""SQLAlchemy 2.0 ORM models for episodic memory — internal, forbidden externally.

Accessed only via :mod:`src.memory.database` package re-exports.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003  # SQLAlchemy needs runtime access

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EpisodeORM(Base):
    """Single episodic memory entry — one interaction or event.

    Renamed from ``Episode`` in phase 5D. The domain-level Episode
    (frozen dataclass) lives in :mod:`src.memory.protocols`. A
    ``Episode = EpisodeORM`` backward-compat alias is re-exported from
    :mod:`src.memory.database`.
    """

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Who was involved
    user_id: Mapped[int] = mapped_column(BigInteger)
    chat_type: Mapped[str] = mapped_column(String(20))
    role: Mapped[str] = mapped_column(String(20))

    # Content
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vector embedding (384 dimensions for paraphrase-multilingual-MiniLM-L12-v2)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384),
        nullable=True,
    )

    # Importance and valence (somatic marker)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    valence: Mapped[str] = mapped_column(String(20), default="neutral")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    # Consolidation status
    consolidated: Mapped[bool] = mapped_column(Boolean, default=False)
    consolidation_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Retrieval tracking (ACT-R frequency component)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Enrichment (Tier 2 background processing)
    enrichment_status: Mapped[str] = mapped_column(String(20), default="pending")
    intent: Mapped[str | None] = mapped_column(String(30), nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(30), nullable=True)
    embedding_version: Mapped[int] = mapped_column(Integer, default=1)

    # Metadata
    source: Mapped[str] = mapped_column(String(50), default="chat")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
