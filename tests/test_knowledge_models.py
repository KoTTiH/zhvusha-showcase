"""Tests for knowledge base SQLAlchemy models."""

from __future__ import annotations

from src.knowledge.models import (
    Category,
    EntryRelation,
    KnowledgeEntry,
    KnowledgeStagingItem,
)


class TestCategory:
    def test_tablename(self) -> None:
        assert Category.__tablename__ == "categories"

    def test_create_minimal(self) -> None:
        cat = Category(name="tech", name_ru="Технологии", path="tech")
        assert cat.name == "tech"
        assert cat.name_ru == "Технологии"
        assert cat.path == "tech"
        assert cat.parent_id is None

    def test_create_with_parent(self) -> None:
        cat = Category(
            name="ai", name_ru="Технологии > AI", path="tech.ai", parent_id=1
        )
        assert cat.parent_id == 1
        assert cat.path == "tech.ai"


class TestKnowledgeEntry:
    def test_tablename(self) -> None:
        assert KnowledgeEntry.__tablename__ == "knowledge_entries"

    def test_create_minimal(self) -> None:
        entry = KnowledgeEntry(title="Test", content="Content")
        assert entry.title == "Test"
        assert entry.content == "Content"
        assert entry.category_id is None
        assert entry.source is None
        assert entry.embedding is None

    def test_create_full(self) -> None:
        entry = KnowledgeEntry(
            title="aiogram 3",
            content="Async Telegram bot framework",
            tags=["python", "telegram"],
            source="manual",
            source_url="https://example.com",
            content_type="tool",
            status="processed",
            embedding=[0.1] * 384,
            token_count=42,
            metadata_={"added_by": "nikita"},
        )
        assert entry.tags == ["python", "telegram"]
        assert entry.content_type == "tool"
        assert entry.status == "processed"
        assert entry.token_count == 42
        assert len(entry.embedding) == 384


class TestEntryRelation:
    def test_tablename(self) -> None:
        assert EntryRelation.__tablename__ == "entry_relations"

    def test_create(self) -> None:
        rel = EntryRelation(source_id=1, target_id=2, relation_type="related")
        assert rel.source_id == 1
        assert rel.target_id == 2
        assert rel.relation_type == "related"


class TestKnowledgeStagingItem:
    def test_tablename(self) -> None:
        assert KnowledgeStagingItem.__tablename__ == "knowledge_staging"

    def test_create(self) -> None:
        item = KnowledgeStagingItem(
            operation="tag",
            target_entry_id=1,
            proposed_changes={"tags": ["python"]},
            reason="Auto-tagged by sleep agent",
            proposed_by="sleep_agent",
        )
        assert item.operation == "tag"
        assert item.proposed_by == "sleep_agent"
