"""Contract tests for the Knowledge Base capability module.

Verifies that ``KnowledgeStore`` conforms to ``KnowledgeStoreProtocol``
at the *contract* level. These tests focus on protocol shape:

* ``runtime_checkable`` ``isinstance`` satisfies the protocol
* every declared protocol method exists on the implementation and is
  callable
* the Pydantic data types from the public contract can be constructed
  and carry their declared defaults
* per-method smoke tests return the contract-declared types (not ORM
  models), using ``mock_session_maker`` from ``tests/conftest.py``

Detailed behaviour coverage (edge cases, SQL semantics, category tree,
staging apply/rollback) lives in ``tests/test_knowledge_store.py``. The
split is intentional: contract tests exercise the public surface of the
capability module, integration tests exercise the implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.knowledge import (
    CategoryInfo,
    FullEntry,
    IndexEntry,
    KnowledgeStore,
    KnowledgeStoreProtocol,
    SearchResult,
    SummaryEntry,
)

if TYPE_CHECKING:
    from src.knowledge.models import KnowledgeEntry

pytestmark = pytest.mark.contract


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def store(mock_session_maker: MagicMock) -> KnowledgeStore:
    """Build a KnowledgeStore backed by the shared mock session maker."""
    return KnowledgeStore(session_maker=mock_session_maker)


@pytest.fixture
def session(mock_session_maker: MagicMock) -> AsyncMock:
    """Expose the mock session owned by ``mock_session_maker``."""
    mock_session: AsyncMock = mock_session_maker._mock_session
    return mock_session


def _make_entry(**fields: Any) -> KnowledgeEntry:
    """Construct a minimal non-ORM-persistent KnowledgeEntry for smoke tests."""
    from src.knowledge.models import KnowledgeEntry as KnowledgeEntryModel

    entry = KnowledgeEntryModel(
        title=fields.get("title", "Example"),
        content=fields.get("content", "Body"),
    )
    entry.id = fields.get("id", 1)
    entry.tags = fields.get("tags", [])
    entry.summary = fields.get("summary")
    entry.source = fields.get("source")
    entry.source_url = fields.get("source_url")
    entry.content_type = fields.get("content_type", "fact")
    entry.status = fields.get("status", "raw")
    entry.token_count = fields.get("token_count", 42)
    entry.category = fields.get("category")
    entry.category_id = fields.get("category_id")
    entry.created_at = None  # type: ignore[assignment]
    entry.updated_at = None  # type: ignore[assignment]
    return entry


# --------------------------------------------------------------------- #
#  Protocol conformance                                                  #
# --------------------------------------------------------------------- #


class TestProtocolConformance:
    """KnowledgeStore must satisfy KnowledgeStoreProtocol structurally."""

    def test_store_is_protocol_instance(self, store: KnowledgeStore) -> None:
        """runtime_checkable isinstance must succeed."""
        assert isinstance(store, KnowledgeStoreProtocol)

    def test_protocol_methods_present_and_callable(self, store: KnowledgeStore) -> None:
        """All 12 protocol methods must exist on the implementation."""
        expected = [
            "hybrid_search",
            "get_index",
            "get_summaries",
            "get_full_content",
            "browse_categories",
            "add_entry",
            "archive_entry",
            "get_or_create_category",
            "propose_change",
            "review_staged",
            "add_relation",
            "recalculate_entry_counts",
        ]
        for name in expected:
            method = getattr(store, name, None)
            assert method is not None, f"Missing protocol method: {name}"
            assert callable(method), f"Protocol method not callable: {name}"


# --------------------------------------------------------------------- #
#  Contract data types                                                   #
# --------------------------------------------------------------------- #


class TestDataTypes:
    """Verify the public Pydantic contract types round-trip and carry defaults."""

    def test_search_result_construction(self) -> None:
        r = SearchResult(id=1, title="t", tags=["a"], rrf_score=0.5)
        assert r.id == 1
        assert r.tags == ["a"]
        assert r.rrf_score == 0.5
        assert r.category_name_ru is None  # default
        assert r.token_count is None  # default

    def test_index_entry_construction(self) -> None:
        e = IndexEntry(id=2, title="i", tags=[])
        assert e.id == 2
        assert e.token_count is None  # default

    def test_summary_entry_construction(self) -> None:
        s = SummaryEntry(id=3, title="s")
        assert s.summary is None  # default

    def test_full_entry_defaults(self) -> None:
        e = FullEntry(id=1, title="T", content="C")
        assert e.content_type == "fact"  # default
        assert e.status == "raw"  # default
        assert e.tags == []  # default
        assert e.source is None  # default

    def test_category_info_defaults(self) -> None:
        c = CategoryInfo(id=1, name="n", name_ru="н", path="x")
        assert c.entry_count == 0  # default
        assert c.summary is None  # default


# --------------------------------------------------------------------- #
#  Per-method smoke tests (return types match the contract)              #
# --------------------------------------------------------------------- #


class TestProtocolMethodSmokes:
    """Smoke tests asserting each protocol method returns its declared type.

    These are *not* exhaustive — the goal is to confirm the contract
    (input shape ⇒ output shape), not to re-test KnowledgeStore's SQL
    semantics. Behaviour details belong in ``tests/test_knowledge_store.py``.
    """

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_hybrid_search_returns_search_results(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        rrf_rows = MagicMock()
        rrf_rows.fetchall = MagicMock(return_value=[(1, 0.42)])
        entries_rows = MagicMock()
        entries_rows.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[_make_entry(id=1)]))
        )
        session.execute = AsyncMock(side_effect=[rrf_rows, entries_rows])

        result = await store.hybrid_search("test query")

        assert isinstance(result, list)
        assert all(isinstance(r, SearchResult) for r in result)

    async def test_get_index_returns_index_entries(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        rows = MagicMock()
        rows.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[_make_entry(id=1)]))
        )
        session.execute = AsyncMock(return_value=rows)

        result = await store.get_index([1])

        assert isinstance(result, list)
        assert all(isinstance(r, IndexEntry) for r in result)

    async def test_get_summaries_returns_summary_entries(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        rows = MagicMock()
        rows.scalars = MagicMock(
            return_value=MagicMock(all=MagicMock(return_value=[_make_entry(id=1)]))
        )
        session.execute = AsyncMock(return_value=rows)

        result = await store.get_summaries([1])

        assert isinstance(result, list)
        assert all(isinstance(r, SummaryEntry) for r in result)

    async def test_get_full_content_returns_full_entry_or_none(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        # Present entry → FullEntry
        session.get = AsyncMock(return_value=_make_entry(id=7))
        present = await store.get_full_content(7)
        assert isinstance(present, FullEntry)
        assert present.id == 7

        # Missing entry → None
        session.get = AsyncMock(return_value=None)
        absent = await store.get_full_content(999)
        assert absent is None

    async def test_browse_categories_returns_category_info(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        rows = MagicMock()
        rows.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        session.execute = AsyncMock(return_value=rows)

        result = await store.browse_categories()

        assert isinstance(result, list)
        assert all(isinstance(c, CategoryInfo) for c in result)

    @pytest.mark.usefixtures("mock_embedding_service")
    async def test_add_entry_returns_int_id(
        self, store: KnowledgeStore, session: AsyncMock
    ) -> None:
        def assign_id(entry: Any) -> None:
            entry.id = 99

        session.add = MagicMock(side_effect=assign_id)

        entry_id = await store.add_entry(title="t", content="c")

        assert isinstance(entry_id, int)
        assert entry_id == 99
