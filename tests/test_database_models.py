"""Tests for Episode SQLAlchemy model and engine factories."""

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from src.memory.database import Base, Episode, get_engine, get_session_maker


def test_episode_has_expected_columns():
    table = Episode.__table__
    column_names = {c.name for c in table.columns}
    expected = {
        "id",
        "timestamp",
        "user_id",
        "chat_type",
        "role",
        "content",
        "summary",
        "embedding",
        "importance",
        "valence",
        "confidence",
        "consolidated",
        "consolidation_result",
        "access_count",
        "last_accessed",
        "enrichment_status",
        "intent",
        "emotion",
        "embedding_version",
        "source",
        "metadata_json",
    }
    assert expected == column_names


def test_episode_column_types():
    table = Episode.__table__
    cols = {c.name: c for c in table.columns}

    assert isinstance(cols["user_id"].type, BigInteger)
    assert isinstance(cols["chat_type"].type, String)
    assert isinstance(cols["role"].type, String)
    assert isinstance(cols["content"].type, Text)
    assert isinstance(cols["importance"].type, Float)
    assert isinstance(cols["valence"].type, String)
    assert isinstance(cols["confidence"].type, Float)
    assert isinstance(cols["consolidated"].type, Boolean)
    assert isinstance(cols["access_count"].type, Integer)
    assert isinstance(cols["timestamp"].type, DateTime)
    assert isinstance(cols["last_accessed"].type, DateTime)
    assert isinstance(cols["source"].type, String)
    assert isinstance(cols["summary"].type, Text)
    assert isinstance(cols["consolidation_result"].type, Text)
    assert isinstance(cols["metadata_json"].type, Text)
    assert isinstance(cols["enrichment_status"].type, String)
    assert isinstance(cols["intent"].type, String)
    assert isinstance(cols["emotion"].type, String)
    assert isinstance(cols["embedding_version"].type, Integer)


def test_episode_defaults():
    table = Episode.__table__
    cols = {c.name: c for c in table.columns}

    assert cols["importance"].default.arg == 0.5
    assert cols["valence"].default.arg == "neutral"
    assert cols["confidence"].default.arg == 0.5
    assert cols["consolidated"].default.arg is False
    assert cols["access_count"].default.arg == 0
    assert cols["source"].default.arg == "chat"
    assert cols["enrichment_status"].default.arg == "pending"
    assert cols["embedding_version"].default.arg == 1


def test_engine_and_session_maker_types():
    engine = get_engine("postgresql+asyncpg://test:test@localhost/test")
    assert isinstance(engine, AsyncEngine)

    maker = get_session_maker(engine)
    assert isinstance(maker, async_sessionmaker)

    # Base should have Episode registered
    assert "episodes" in Base.metadata.tables
