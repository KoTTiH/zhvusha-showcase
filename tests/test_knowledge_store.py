"""Tests for KnowledgeStore data access layer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.knowledge.models import KnowledgeEntry, KnowledgeStagingItem
from src.knowledge.store import (
    CategoryInfo,
    FullEntry,
    IndexEntry,
    KnowledgeStore,
    SearchResult,
    SummaryEntry,
)


@pytest.fixture
def store(mock_session_maker: MagicMock) -> KnowledgeStore:
    return KnowledgeStore(session_maker=mock_session_maker)


@pytest.fixture
def session(mock_session_maker: MagicMock) -> AsyncMock:
    return mock_session_maker._mock_session


def _make_entry(
    entry_id: int = 1,
    title: str = "Test Entry",
    content: str = "Test content",
    **kwargs: Any,
) -> KnowledgeEntry:
    entry = KnowledgeEntry(title=title, content=content, **kwargs)
    entry.id = entry_id
    entry.tags = kwargs.get("tags", [])
    entry.token_count = kwargs.get("token_count", 42)
    entry.summary = kwargs.get("summary", "Test summary")
    entry.source = kwargs.get("source")
    entry.source_url = kwargs.get("source_url")
    entry.content_type = kwargs.get("content_type", "fact")
    entry.status = kwargs.get("status", "raw")
    entry.category = kwargs.get("category")
    entry.created_at = None
    entry.updated_at = None
    return entry


class TestAddEntry:
    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_add_entry_returns_id(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # Simulate flush setting ID
        def set_id(entry: Any) -> None:
            entry.id = 42

        session.add = MagicMock(side_effect=set_id)

        entry_id = await store.add_entry(
            title="Test",
            content="Content",
            tags=["python"],
            source="manual",
        )
        assert entry_id == 42
        session.commit.assert_awaited_once()

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_add_entry_with_category(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # Mock category resolution
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=5)
        session.execute = AsyncMock(return_value=mock_result)

        def set_id(entry: Any) -> None:
            entry.id = 1

        session.add = MagicMock(side_effect=set_id)

        entry_id = await store.add_entry(
            title="Test",
            content="Content",
            category_path="tools.python",
        )
        assert entry_id == 1


class TestGetEntry:
    async def test_get_existing(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        expected = _make_entry(entry_id=1)
        session.get = AsyncMock(return_value=expected)

        result = await store.get_entry(1)
        assert result is not None
        assert result.title == "Test Entry"

    async def test_get_nonexistent(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        session.get = AsyncMock(return_value=None)

        result = await store.get_entry(999)
        assert result is None


class TestUpdateEntry:
    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_update_fields(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        ok = await store.update_entry(1, status="verified")
        assert ok is True
        session.commit.assert_awaited()

    async def test_update_empty_fields(self, store: KnowledgeStore) -> None:
        ok = await store.update_entry(1)
        assert ok is False

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_update_content_re_embeds(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # First call returns existing entry, second call does the update
        existing = _make_entry(entry_id=1)
        session.get = AsyncMock(return_value=existing)

        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        ok = await store.update_entry(1, content="New content")
        assert ok is True


class TestArchiveEntry:
    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_archive(self, store: KnowledgeStore, session: AsyncMock) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        ok = await store.archive_entry(1)
        assert ok is True


class TestProgressiveDisclosure:
    async def test_get_index(self, store: KnowledgeStore, session: AsyncMock) -> None:
        entries = [_make_entry(1, tags=["a"]), _make_entry(2, tags=["b"])]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        index = await store.get_index([1, 2])
        assert len(index) == 2
        assert all(isinstance(e, IndexEntry) for e in index)

    async def test_get_summaries(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entries = [_make_entry(1, summary="Sum 1"), _make_entry(2, summary="Sum 2")]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        summaries = await store.get_summaries([1, 2])
        assert len(summaries) == 2
        assert all(isinstance(s, SummaryEntry) for s in summaries)
        assert summaries[0].summary == "Sum 1"

    async def test_get_full_content(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entry = _make_entry(1, content="Full text here")
        session.get = AsyncMock(return_value=entry)

        full = await store.get_full_content(1)
        assert full is not None
        assert isinstance(full, FullEntry)
        assert full.content == "Full text here"

    async def test_get_full_content_not_found(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        session.get = AsyncMock(return_value=None)

        full = await store.get_full_content(999)
        assert full is None


class TestCategories:
    async def test_browse_root(self, store: KnowledgeStore, session: AsyncMock) -> None:
        cats = [
            SimpleNamespace(
                id=1,
                name="tech",
                name_ru="Технологии",
                path="tech",
                entry_count=10,
                summary="Tech stuff",
            ),
        ]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=cats))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.browse_categories()
        assert len(result) == 1
        assert isinstance(result[0], CategoryInfo)
        assert result[0].name_ru == "Технологии"

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_get_or_create_existing(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # ON CONFLICT DO NOTHING → scalar_one_or_none returns None (no insert)
        # Then SELECT → scalar_one returns 5
        upsert_result = MagicMock()
        upsert_result.scalar_one_or_none = MagicMock(return_value=None)
        select_result = MagicMock()
        select_result.scalar_one = MagicMock(return_value=5)
        session.execute = AsyncMock(side_effect=[upsert_result, select_result])

        cat_id = await store.get_or_create_category("tech", "Технологии")
        assert cat_id == 5
        session.rollback.assert_not_awaited()

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_get_or_create_new(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # INSERT succeeds → RETURNING gives us the new id directly
        insert_result = MagicMock()
        insert_result.scalar_one_or_none = MagicMock(return_value=10)
        session.execute = AsyncMock(return_value=insert_result)

        cat_id = await store.get_or_create_category("tech", "Технологии")
        assert cat_id == 10
        session.commit.assert_awaited_once()


class TestStaging:
    async def test_propose_change(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        def set_id(item: Any) -> None:
            item.id = 7

        session.add = MagicMock(side_effect=set_id)

        staging_id = await store.propose_change(
            operation="tag",
            target_entry_id=1,
            proposed_changes={"tags": ["python"]},
            reason="Auto-tag",
            proposed_by="sleep_agent",
        )
        assert staging_id == 7

    async def test_review_staged_approve_applies_add(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "add"
        staging_item.target_entry_id = None
        staging_item.proposed_changes = {
            "title": "Test",
            "content": "Content",
            "source": "mcp_server",
        }
        session.get = AsyncMock(return_value=staging_item)

        with patch.object(
            store, "add_entry", new_callable=AsyncMock, return_value=42
        ) as mock_add:
            ok = await store.review_staged(7, approve=True)

        assert ok is True
        mock_add.assert_awaited_once_with(
            title="Test",
            content="Content",
            source="mcp_server",
            category_path=None,
            tags=None,
        )

    async def test_review_staged_approve_applies_tag(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "tag"
        staging_item.target_entry_id = 5
        staging_item.proposed_changes = {"tags": ["python", "ai"]}
        session.get = AsyncMock(return_value=staging_item)

        with patch.object(
            store, "update_entry", new_callable=AsyncMock, return_value=True
        ) as mock_upd:
            ok = await store.review_staged(7, approve=True)

        assert ok is True
        mock_upd.assert_awaited_once_with(5, tags=["python", "ai"])

    async def test_review_staged_approve_applies_summarize(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "summarize"
        staging_item.target_entry_id = 3
        staging_item.proposed_changes = {"summary": "Краткое описание."}
        session.get = AsyncMock(return_value=staging_item)

        with patch.object(
            store, "update_entry", new_callable=AsyncMock, return_value=True
        ) as mock_upd:
            ok = await store.review_staged(8, approve=True)

        assert ok is True
        mock_upd.assert_awaited_once_with(3, summary="Краткое описание.")

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_review_staged_approve_applies_categorize(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "categorize"
        staging_item.target_entry_id = 4
        staging_item.proposed_changes = {"category_path": "tools.python"}
        session.get = AsyncMock(return_value=staging_item)

        with (
            patch.object(
                store,
                "get_or_create_category",
                new_callable=AsyncMock,
                return_value=10,
            ) as mock_cat,
            patch.object(
                store, "update_entry", new_callable=AsyncMock, return_value=True
            ) as mock_upd,
        ):
            ok = await store.review_staged(9, approve=True)

        assert ok is True
        mock_cat.assert_awaited_once_with("tools.python", "Tools > Python")
        mock_upd.assert_awaited_once_with(4, category_id=10)

    async def test_review_staged_reject(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        ok = await store.review_staged(7, approve=False)
        assert ok is True

    async def test_review_staged_not_found(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)

        ok = await store.review_staged(999, approve=True)
        assert ok is False

    async def test_review_staged_apply_failure_rolls_back_status(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        """If _apply_staging_item raises, status must revert to pending."""
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "add"
        staging_item.target_entry_id = None
        staging_item.proposed_changes = {
            "title": "Fail",
            "content": "Content",
        }
        session.get = AsyncMock(return_value=staging_item)

        with (
            patch.object(
                store,
                "add_entry",
                new_callable=AsyncMock,
                side_effect=RuntimeError("embed down"),
            ),
            pytest.raises(RuntimeError, match="embed down"),
        ):
            await store.review_staged(7, approve=True)

        # Verify rollback: last execute call should set status back to pending
        last_call = session.execute.await_args_list[-1]
        stmt = last_call.args[0]
        # The compiled statement should contain 'pending'
        compiled = stmt.compile(
            compile_kwargs={"literal_binds": True},
        )
        assert "pending" in str(compiled)

    async def test_review_staged_apply_invalid_add_skips(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        """operation=add without title/content should not crash."""
        mock_update = MagicMock()
        mock_update.rowcount = 1
        session.execute = AsyncMock(return_value=mock_update)

        staging_item = MagicMock(spec=KnowledgeStagingItem)
        staging_item.operation = "add"
        staging_item.target_entry_id = None
        staging_item.proposed_changes = {"source": "bad_data"}  # no title/content
        session.get = AsyncMock(return_value=staging_item)

        ok = await store.review_staged(10, approve=True)
        assert ok is True  # no crash, graceful skip


class TestQueryHelpers:
    """Tests for get_untagged, get_unsummarized, get_uncategorized."""

    async def test_get_untagged_returns_entries_with_empty_tags(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entries = [_make_entry(1, tags=[]), _make_entry(2, tags=[])]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_untagged(limit=5)
        assert len(result) == 2
        assert all(e.tags == [] for e in result)

    async def test_get_untagged_respects_limit(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entries = [_make_entry(1, tags=[])]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_untagged(limit=1)
        assert len(result) == 1

    async def test_get_untagged_empty(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_untagged()
        assert result == []

    async def test_get_unsummarized_returns_entries_without_summary(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entries = [_make_entry(1, summary=None), _make_entry(2, summary=None)]
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_unsummarized(limit=5)
        assert len(result) == 2
        assert all(e.summary is None for e in result)

    async def test_get_unsummarized_empty(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_unsummarized()
        assert result == []

    async def test_get_uncategorized_returns_entries_without_category(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        entries = [
            _make_entry(1, category=None),
            _make_entry(2, category=None),
        ]
        # Ensure category_id is None
        for e in entries:
            e.category_id = None
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=entries))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_uncategorized(limit=5)
        assert len(result) == 2

    async def test_get_uncategorized_empty(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[]))
        )
        session.execute = AsyncMock(return_value=mock_result)

        result = await store.get_uncategorized()
        assert result == []


class TestRelations:
    async def test_add_relation(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        def set_id(rel: Any) -> None:
            rel.id = 3

        session.add = MagicMock(side_effect=set_id)

        rel_id = await store.add_relation(1, 2, "related")
        assert rel_id == 3


class TestSearchResult:
    def test_model_fields(self) -> None:
        r = SearchResult(
            id=1,
            title="T",
            tags=["a"],
            rrf_score=0.5,
            category_name_ru="Cat",
            token_count=100,
        )
        assert r.rrf_score == 0.5
        assert r.category_name_ru == "Cat"
