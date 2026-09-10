"""Knowledge base data access layer.

Central class for CRUD, hybrid search (BM25 + semantic + RRF),
progressive disclosure, and staging. Used by bot, MCP server, and daemon.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog
from sqlalchemy import CursorResult, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.embeddings import EmbeddingService
from src.knowledge.models import (
    Category,
    EntryRelation,
    KnowledgeEntry,
    KnowledgeStagingItem,
)

# Re-export Pydantic data types from the public contract so that existing
# imports `from src.knowledge.store import SearchResult, …` keep working
# during phase 4 (tests and the MCP server rely on this shim). The single
# source of truth now lives in src/knowledge/protocols.py — see `__all__`
# below for the list of public names.
from src.knowledge.protocols import (
    CategoryInfo,
    FullEntry,
    IndexEntry,
    KnowledgeStoreProtocol,
    SearchResult,
    SummaryEntry,
)

__all__ = [
    "CategoryInfo",
    "FullEntry",
    "IndexEntry",
    "KnowledgeStore",
    "KnowledgeStoreProtocol",
    "SearchResult",
    "SummaryEntry",
]

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = structlog.get_logger()


# --- Hybrid search SQL ---

_HYBRID_SEARCH_SQL = text("""\
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> :query_embedding) AS rank_s
    FROM knowledge_entries
    WHERE status != 'archived'
      AND embedding IS NOT NULL
    ORDER BY embedding <=> :query_embedding
    LIMIT 30
),
lexical AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank(
            to_tsvector('russian', coalesce(title,'') || ' ' || coalesce(content,'')),
            plainto_tsquery('russian', :query_text)
        ) DESC
    ) AS rank_l
    FROM knowledge_entries
    WHERE status != 'archived'
      AND to_tsvector('russian', coalesce(title,'') || ' ' || coalesce(content,''))
          @@ plainto_tsquery('russian', :query_text)
    LIMIT 30
)
SELECT
    COALESCE(s.id, l.id) AS id,
    (1.0 / (60 + COALESCE(s.rank_s, 1000)))
    + (1.0 / (60 + COALESCE(l.rank_l, 1000))) AS rrf_score
FROM semantic s
FULL OUTER JOIN lexical l ON s.id = l.id
ORDER BY rrf_score DESC
LIMIT :result_limit
""")


class KnowledgeStore(KnowledgeStoreProtocol):
    """Data access layer for the knowledge base.

    Provides CRUD, hybrid search, progressive disclosure, category
    navigation, staging, and relations.

    Formally implements :class:`KnowledgeStoreProtocol` for the clean
    subset of methods. ORM-leaking methods (``get_entry``,
    ``get_untagged``, ``get_unsummarized``, ``get_uncategorized``,
    ``get_pending_staged``, ``update_entry``) and the ``session()``
    escape hatch are intentionally not part of the protocol — they
    remain accessible on this concrete class for clients that need
    them (daemon ``sleep_agent``, MCP dashboard_api).
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_maker = session_maker

    def session(self) -> AsyncSession:
        """Create a new async session. Must be used as ``async with store.session()``."""
        return self._session_maker()

    # ------------------------------------------------------------------ #
    #  CRUD                                                                #
    # ------------------------------------------------------------------ #

    async def add_entry(
        self,
        title: str,
        content: str,
        *,
        category_path: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        source_url: str | None = None,
        content_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new knowledge entry. Returns its ID."""
        embedding = await EmbeddingService.embed_async(f"{title} {content[:500]}")
        token_count = len(content.split()) * 4 // 3  # rough estimate

        category_id: int | None = None
        if category_path:
            # Create category tree if it doesn't exist
            name_ru = " > ".join(
                part.replace("_", " ").capitalize() for part in category_path.split(".")
            )
            category_id = await self.get_or_create_category(category_path, name_ru)

        entry = KnowledgeEntry(
            title=title,
            content=content,
            category_id=category_id,
            tags=tags or [],
            source=source,
            source_url=source_url,
            content_type=content_type,
            embedding=embedding,
            token_count=token_count,
            metadata_=metadata or {},
        )

        async with self._session_maker() as session:
            session.add(entry)
            await session.flush()
            entry_id: int = entry.id
            await session.commit()

        if category_id is not None:
            await self._increment_entry_count(category_id)

        logger.info("knowledge_entry_added", id=entry_id, title=title)
        return entry_id

    async def get_entry(self, entry_id: int) -> KnowledgeEntry | None:
        """Get a single entry by ID."""
        async with self._session_maker() as session:
            return await session.get(KnowledgeEntry, entry_id)

    async def update_entry(self, entry_id: int, **fields: Any) -> bool:
        """Update entry fields. Re-embeds if title/content changed."""
        if not fields:
            return False

        # Re-embed if content changed
        if "title" in fields or "content" in fields:
            async with self._session_maker() as session:
                entry = await session.get(KnowledgeEntry, entry_id)
                if entry is None:
                    return False
                title = fields.get("title", entry.title)
                content = fields.get("content", entry.content)
            fields["embedding"] = await EmbeddingService.embed_async(
                f"{title} {content[:500]}"
            )
            fields["token_count"] = len(content.split()) * 4 // 3

        fields["updated_at"] = datetime.now(tz=UTC)

        async with self._session_maker() as session:
            result = await session.execute(
                update(KnowledgeEntry)
                .where(KnowledgeEntry.id == entry_id)
                .values(**fields)
            )
            await session.commit()
            cursor = cast("CursorResult[Any]", result)
            return cursor.rowcount > 0

    async def archive_entry(self, entry_id: int) -> bool:
        """Set entry status to 'archived' and decrement category counter."""
        # Get category_id before archiving
        entry = await self.get_entry(entry_id)
        if entry is None:
            return False
        category_id = entry.category_id
        old_status = entry.status

        result = await self.update_entry(entry_id, status="archived")
        if result and category_id is not None and old_status != "archived":
            await self._decrement_entry_count(category_id)
        return result

    # ------------------------------------------------------------------ #
    #  Hybrid Search (BM25 + Semantic + RRF)                               #
    # ------------------------------------------------------------------ #

    async def hybrid_search(
        self,
        query: str,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search knowledge base using Reciprocal Rank Fusion."""
        query_embedding = await EmbeddingService.embed_async(query)

        async with self._session_maker() as session:
            # Execute hybrid RRF query
            rows = await session.execute(
                _HYBRID_SEARCH_SQL,
                {
                    "query_embedding": str(query_embedding),
                    "query_text": query,
                    "result_limit": limit * 2,  # fetch extra for filtering
                },
            )
            rrf_results = list(rows.fetchall())

        if not rrf_results:
            return []

        # Fetch entry details for matched IDs
        matched_ids = [r[0] for r in rrf_results]
        score_map = {r[0]: float(r[1]) for r in rrf_results}

        async with self._session_maker() as session:
            stmt = select(KnowledgeEntry).where(KnowledgeEntry.id.in_(matched_ids))
            result = await session.execute(stmt)
            entries = list(result.scalars().all())

        # Post-filter by category and tags
        results: list[SearchResult] = []
        for entry in entries:
            if category and (
                not entry.category or not entry.category.path.startswith(category)
            ):
                continue
            if tags and not set(tags).intersection(set(entry.tags)):
                continue

            results.append(
                SearchResult(
                    id=entry.id,
                    title=entry.title,
                    tags=entry.tags,
                    category_name_ru=(
                        entry.category.name_ru if entry.category else None
                    ),
                    token_count=entry.token_count,
                    rrf_score=score_map.get(entry.id, 0.0),
                )
            )

        results.sort(key=lambda r: r.rrf_score, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------ #
    #  Progressive Disclosure                                              #
    # ------------------------------------------------------------------ #

    async def get_index(self, entry_ids: list[int]) -> list[IndexEntry]:
        """Level 1: index entries (~20 tokens each)."""
        async with self._session_maker() as session:
            stmt = select(KnowledgeEntry).where(KnowledgeEntry.id.in_(entry_ids))
            result = await session.execute(stmt)
            entries = result.scalars().all()

        return [
            IndexEntry(
                id=e.id,
                title=e.title,
                tags=e.tags,
                token_count=e.token_count,
            )
            for e in entries
        ]

    async def get_summaries(self, entry_ids: list[int]) -> list[SummaryEntry]:
        """Level 2: summaries (~100 tokens each)."""
        async with self._session_maker() as session:
            stmt = select(KnowledgeEntry).where(KnowledgeEntry.id.in_(entry_ids))
            result = await session.execute(stmt)
            entries = result.scalars().all()

        return [
            SummaryEntry(
                id=e.id,
                title=e.title,
                summary=e.summary,
            )
            for e in entries
        ]

    async def get_full_content(self, entry_id: int) -> FullEntry | None:
        """Level 3: full text of a single entry."""
        async with self._session_maker() as session:
            entry = await session.get(KnowledgeEntry, entry_id)
            if entry is None:
                return None

            return FullEntry(
                id=entry.id,
                title=entry.title,
                content=entry.content,
                summary=entry.summary,
                tags=entry.tags,
                source=entry.source,
                source_url=entry.source_url,
                content_type=entry.content_type,
                status=entry.status,
                category_name_ru=(entry.category.name_ru if entry.category else None),
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )

    # ------------------------------------------------------------------ #
    #  Categories                                                          #
    # ------------------------------------------------------------------ #

    async def browse_categories(
        self, parent_path: str | None = None
    ) -> list[CategoryInfo]:
        """List categories. Root if no parent_path, children otherwise."""
        async with self._session_maker() as session:
            if parent_path is None:
                # Root categories: those without parent
                stmt = select(Category).where(Category.parent_id.is_(None))
            else:
                # Direct children of parent_path
                stmt = select(Category).where(
                    Category.parent_id
                    == select(Category.id)
                    .where(Category.path == parent_path)
                    .correlate(None)
                    .scalar_subquery()
                )

            result = await session.execute(stmt)
            categories = result.scalars().all()

        return [
            CategoryInfo(
                id=c.id,
                name=c.name,
                name_ru=c.name_ru,
                path=c.path,
                entry_count=c.entry_count,
                summary=c.summary,
            )
            for c in categories
        ]

    async def get_or_create_category(self, path: str, name_ru: str) -> int:
        """Get category by path or create it. Returns category ID.

        Uses INSERT ON CONFLICT to avoid TOCTOU race between SELECT and INSERT.
        """
        # Determine parent first (recursive)
        parent_id: int | None = None
        parts = path.split(".")
        if len(parts) > 1:
            parent_path = ".".join(parts[:-1])
            parent_name_ru = " > ".join(name_ru.split(" > ")[:-1]) or parent_path
            parent_id = await self.get_or_create_category(parent_path, parent_name_ru)

        name = parts[-1]

        async with self._session_maker() as session:
            # Atomic upsert: insert or return existing
            insert_stmt = pg_insert(Category).values(
                name=name,
                name_ru=name_ru,
                path=path,
                parent_id=parent_id,
            )
            upsert_stmt = insert_stmt.on_conflict_do_nothing(
                index_elements=["path"],
            ).returning(Category.id)

            result = await session.execute(upsert_stmt)
            row = result.scalar_one_or_none()

            if row is not None:
                await session.commit()
                logger.info("category_created", path=path, id=row)
                return row

            # Row already existed — fetch it
            select_stmt = select(Category.id).where(Category.path == path)
            result = await session.execute(select_stmt)
            cat_id: int = result.scalar_one()
            return cat_id

    # ------------------------------------------------------------------ #
    #  Query helpers (used by SleepTimeAgent)                              #
    # ------------------------------------------------------------------ #

    async def get_untagged(self, limit: int = 5) -> list[KnowledgeEntry]:
        """Get non-archived entries with empty tag lists."""
        async with self._session_maker() as session:
            stmt = (
                select(KnowledgeEntry)
                .where(KnowledgeEntry.tags == [])
                .where(KnowledgeEntry.status != "archived")
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_unsummarized(self, limit: int = 5) -> list[KnowledgeEntry]:
        """Get non-archived entries without summaries."""
        async with self._session_maker() as session:
            stmt = (
                select(KnowledgeEntry)
                .where(KnowledgeEntry.summary.is_(None))
                .where(KnowledgeEntry.status != "archived")
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_uncategorized(self, limit: int = 5) -> list[KnowledgeEntry]:
        """Get non-archived entries without categories."""
        async with self._session_maker() as session:
            stmt = (
                select(KnowledgeEntry)
                .where(KnowledgeEntry.category_id.is_(None))
                .where(KnowledgeEntry.status != "archived")
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    #  Staging                                                             #
    # ------------------------------------------------------------------ #

    async def propose_change(
        self,
        operation: str,
        target_entry_id: int | None,
        proposed_changes: dict[str, Any],
        reason: str,
        proposed_by: str,
    ) -> int:
        """Create a staging proposal. Returns staging item ID."""
        item = KnowledgeStagingItem(
            operation=operation,
            target_entry_id=target_entry_id,
            proposed_changes=proposed_changes,
            reason=reason,
            proposed_by=proposed_by,
        )

        async with self._session_maker() as session:
            session.add(item)
            await session.flush()
            item_id: int = item.id
            await session.commit()

        return item_id

    async def review_staged(self, staging_id: int, approve: bool) -> bool:
        """Approve or reject a staging proposal. Approved items are applied."""
        new_status = "approved" if approve else "rejected"

        async with self._session_maker() as session:
            # Atomic update — only one caller can win the pending→approved race
            result = await session.execute(
                update(KnowledgeStagingItem)
                .where(KnowledgeStagingItem.id == staging_id)
                .where(KnowledgeStagingItem.status == "pending")
                .values(status=new_status)
            )
            await session.commit()
            cursor = cast("CursorResult[Any]", result)
            if cursor.rowcount == 0:
                return False

        if not approve:
            return True

        # Fetch approved item and apply changes
        async with self._session_maker() as session:
            item = await session.get(KnowledgeStagingItem, staging_id)

        if item is not None and item.proposed_changes:
            try:
                await self._apply_staging_item(
                    item.operation, item.target_entry_id, item.proposed_changes
                )
            except Exception:
                logger.exception(
                    "staging_apply_failed",
                    staging_id=staging_id,
                    operation=item.operation,
                )
                # Rollback status so the item can be retried
                async with self._session_maker() as session:
                    await session.execute(
                        update(KnowledgeStagingItem)
                        .where(KnowledgeStagingItem.id == staging_id)
                        .values(status="pending")
                    )
                    await session.commit()
                raise

        return True

    async def _apply_staging_item(
        self,
        operation: str,
        target_entry_id: int | None,
        proposed_changes: dict[str, Any],
    ) -> None:
        """Apply an approved staging change to the knowledge base."""
        if operation == "add":
            if "title" not in proposed_changes or "content" not in proposed_changes:
                logger.warning(
                    "staging_apply_invalid",
                    operation=operation,
                    reason="missing required fields: title, content",
                )
                return
            await self.add_entry(
                title=proposed_changes["title"],
                content=proposed_changes["content"],
                category_path=proposed_changes.get("category_path"),
                tags=proposed_changes.get("tags"),
                source=proposed_changes.get("source", "staging"),
            )
        elif target_entry_id is not None:
            fields: dict[str, Any] = {}
            if "tags" in proposed_changes:
                fields["tags"] = proposed_changes["tags"]
            if "summary" in proposed_changes:
                fields["summary"] = proposed_changes["summary"]
            if "category_path" in proposed_changes:
                cat_path = proposed_changes["category_path"]
                name_ru = " > ".join(
                    part.replace("_", " ").capitalize() for part in cat_path.split(".")
                )
                cat_id = await self.get_or_create_category(cat_path, name_ru)
                fields["category_id"] = cat_id
            if fields:
                await self.update_entry(target_entry_id, **fields)
        else:
            logger.warning(
                "staging_apply_skip",
                operation=operation,
                reason="unknown operation or missing target_entry_id",
            )

    async def get_pending_staged(self, limit: int = 50) -> list[KnowledgeStagingItem]:
        """Get pending staging proposals for review."""
        async with self._session_maker() as session:
            stmt = (
                select(KnowledgeStagingItem)
                .where(KnowledgeStagingItem.status == "pending")
                .order_by(KnowledgeStagingItem.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    #  Relations                                                           #
    # ------------------------------------------------------------------ #

    async def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
    ) -> int:
        """Add a directed relation between entries."""
        rel = EntryRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
        )

        async with self._session_maker() as session:
            session.add(rel)
            await session.flush()
            rel_id: int = rel.id
            await session.commit()

        return rel_id

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    async def _resolve_category(self, path: str) -> int | None:
        """Resolve category path to ID. Returns None if not found."""
        async with self._session_maker() as session:
            stmt = select(Category.id).where(Category.path == path)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row

    async def _increment_entry_count(self, category_id: int) -> None:
        """Increment entry_count for a category."""
        async with self._session_maker() as session:
            await session.execute(
                update(Category)
                .where(Category.id == category_id)
                .values(entry_count=Category.entry_count + 1)
            )
            await session.commit()

    async def _decrement_entry_count(self, category_id: int) -> None:
        """Decrement entry_count for a category (floor at 0)."""
        async with self._session_maker() as session:
            await session.execute(
                update(Category)
                .where(Category.id == category_id)
                .where(Category.entry_count > 0)
                .values(entry_count=Category.entry_count - 1)
            )
            await session.commit()

    async def recalculate_entry_counts(self) -> None:
        """Recalculate all category entry_counts from actual data."""
        async with self._session_maker() as session:
            # Reset all to 0
            await session.execute(update(Category).values(entry_count=0))
            # Count non-archived entries per category
            counts = await session.execute(
                select(
                    KnowledgeEntry.category_id,
                    func.count(KnowledgeEntry.id),
                )
                .where(
                    KnowledgeEntry.status != "archived",
                    KnowledgeEntry.category_id.isnot(None),
                )
                .group_by(KnowledgeEntry.category_id)
            )
            for cat_id, count in counts.fetchall():
                await session.execute(
                    update(Category)
                    .where(Category.id == cat_id)
                    .values(entry_count=count)
                )
            await session.commit()
        logger.info("entry_counts_recalculated")
