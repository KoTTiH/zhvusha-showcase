"""Episodic memory — fast-learning hippocampal store.

Records episodes with one-shot learning. Retrieves by combined ACT-R
hybrid score: BLA(frequency+recency) + cosine_similarity + importance.
Implements somatic markers, pattern separation, and pattern completion.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select, update

from src.embeddings import EmbeddingService
from src.memory.database import EpisodeORM
from src.memory.protocols import Episode as EpisodeDomain
from src.memory.protocols import EpisodicMemoryProtocol

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.memory.types import EnrichmentResult

# Social mode rate limit: max episodes per user per hour
_SOCIAL_RATE_LIMIT = 10
_SOCIAL_RATE_WINDOW = 3600.0  # 1 hour in seconds

logger = logging.getLogger(__name__)


def _build_merged_metadata_json(
    current: str | None,
    enrichment_updates: dict[str, object],
) -> str:
    """Merge enrichment fields into existing `metadata_json`, preserving
    any top-level keys that aren't under the `enrichment` sub-key.

    Returns valid JSON always. Corrupted or non-dict input is discarded
    (the row is about to be overwritten anyway, so losing malformed
    metadata is acceptable).

    Note: this is the *correct* merge. The legacy Tier 2 path at
    `src/memory/enrichment.py:160` unconditionally overwrites top-level
    keys — a latent bug. Do not replicate it. Until that path is fixed,
    `update_enrichment` is the sole writer of enrichment metadata; if a
    second writer is introduced, this merge must become atomic (JSONB
    column + jsonb_set).
    """
    meta: dict[str, object] = {}
    if current:
        try:
            parsed = json.loads(current)
            if isinstance(parsed, dict):
                meta = parsed
        except json.JSONDecodeError:
            pass  # corrupted: start from empty

    existing_enrichment = meta.get("enrichment")
    if not isinstance(existing_enrichment, dict):
        existing_enrichment = {}
    existing_enrichment.update(enrichment_updates)
    meta["enrichment"] = existing_enrichment
    return json.dumps(meta, ensure_ascii=False)


class EpisodicMemory(EpisodicMemoryProtocol):
    """Fast-learning hippocampal store.

    Records episodes with one-shot learning.
    Retrieves by ACT-R hybrid score.

    Implements :class:`EpisodicMemoryProtocol`. Public read methods return
    :class:`EpisodeDomain` (frozen dataclass). Internal storage uses
    :class:`EpisodeORM` (SQLAlchemy). Conversion happens in
    ``_orm_to_domain``.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        admin_user_id: int,
    ) -> None:
        self.session_maker = session_maker
        self.admin_user_id = admin_user_id
        # Social rate limit: {user_id: [timestamps]}
        self._social_timestamps: dict[int, list[float]] = {}

    @staticmethod
    def _orm_to_domain(orm: EpisodeORM) -> EpisodeDomain:
        """Convert SQLAlchemy ORM Episode to frozen domain Episode.

        Field parity between the two is enforced by a contract test.
        """
        return EpisodeDomain(
            **{f.name: getattr(orm, f.name) for f in dataclasses.fields(EpisodeDomain)}
        )

    async def record(
        self,
        content: str,
        user_id: int,
        chat_type: str,
        role: str,
        importance: float = 0.5,
        valence: str = "neutral",
        confidence: float = 0.5,
        source: str = "chat",
        metadata: dict[str, object] | None = None,
        person_name: str = "unknown",
        significance: str = "stranger",
        domain: str = "chat",
    ) -> int:
        """Record a new episode. Returns episode ID.

        Recording rules by chat_type:
        - personal: full content, Tier 1 enriched embedding
        - assistant: full content, Tier 1 enriched embedding, base importance 0.3
        - social: summary only (first 100 chars), no embedding, importance 0.1,
          rate limited to 10/user/hour
        """
        # Social rate limiting
        if chat_type == "social":
            if not self._check_social_rate(user_id):
                return -1
            content = content[:100]
            importance = 0.1

        # Build Tier 1 enriched embed text
        embedding: list[float] | None = None
        if chat_type != "social":
            embed_text = self._build_tier1_embed_text(
                content=content,
                source=source,
                chat_type=chat_type,
                role=role,
                person_name=person_name,
                significance=significance,
                domain=domain,
                importance=importance,
            )
            try:
                embedding = await EmbeddingService.embed_async(embed_text)
            except Exception as exc:
                logger.warning(
                    "episodic_embedding_failed",
                    extra={"error_type": type(exc).__name__},
                )

        episode = EpisodeORM(
            user_id=user_id,
            chat_type=chat_type,
            role=role,
            content=content,
            embedding=embedding,
            importance=importance,
            valence=valence,
            confidence=confidence,
            source=source,
            metadata_json=json.dumps(metadata) if metadata else None,
        )

        async with self.session_maker() as session:
            session.add(episode)
            await session.flush()
            episode_id: int = episode.id
            await session.commit()

        return episode_id

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        chat_type: str = "personal",
        source_filter: list[str] | None = None,
    ) -> list[EpisodeDomain]:
        """Retrieve episodes by ACT-R hybrid score.

        Two-stage retrieval:
        1. pgvector cosine distance → top-50 candidates
        2. Python ACT-R scoring → final top-N

        Score = BLA + 2.0 * cosine_sim + 1.5 * (importance - 0.5)
        BLA = ln(n/0.5) - 0.5*ln(L) + ln(1 + t_recent^(-0.5))

        Args:
            source_filter: If set, only return episodes with source IN this list.
        """
        query_embedding = await EmbeddingService.embed_async(query)
        now = datetime.now(tz=UTC)

        async with self.session_maker() as session:
            # Stage 1: pgvector cosine distance pre-filter
            stmt = (
                select(EpisodeORM)
                .where(EpisodeORM.embedding.isnot(None))
                .order_by(EpisodeORM.embedding.cosine_distance(query_embedding))
                .limit(100)
            )
            if source_filter:
                stmt = stmt.where(EpisodeORM.source.in_(source_filter))
            result = await session.execute(stmt)
            candidates = list(result.scalars().all())

            if not candidates:
                return []

            # Stage 2: ACT-R hybrid scoring
            scored: list[tuple[EpisodeORM, float]] = []
            for ep in candidates:
                score = self._act_r_score(ep, query_embedding, now)
                scored.append((ep, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:limit]

            # Strengthen neural pathways: increment access_count
            episode_ids = [ep.id for ep, _ in top]
            if episode_ids:
                await session.execute(
                    update(EpisodeORM)
                    .where(EpisodeORM.id.in_(episode_ids))
                    .values(
                        access_count=EpisodeORM.access_count + 1,
                        last_accessed=now,
                    )
                )
                await session.commit()

        return [self._orm_to_domain(ep) for ep, _ in top]

    async def retrieve_by_somatic_marker(
        self,
        query: str,
        limit: int = 3,
    ) -> list[tuple[EpisodeDomain, float]]:
        """System 1 fast path: find similar episodes with their valence.

        Only returns episodes with confidence > 0.5.
        Returns (episode, cosine_similarity) tuples.
        """
        query_embedding = await EmbeddingService.embed_async(query)

        async with self.session_maker() as session:
            stmt = (
                select(EpisodeORM)
                .where(
                    EpisodeORM.embedding.isnot(None),
                    EpisodeORM.confidence > 0.5,
                )
                .order_by(EpisodeORM.embedding.cosine_distance(query_embedding))
                .limit(limit)
            )
            result = await session.execute(stmt)
            episodes = list(result.scalars().all())

        results: list[tuple[EpisodeDomain, float]] = []
        for ep in episodes:
            if ep.embedding is not None:
                sim = EmbeddingService.cosine_similarity(query_embedding, ep.embedding)
                results.append((self._orm_to_domain(ep), sim))

        return results

    async def complete_pattern(
        self,
        partial_cue: str,
        threshold: float = 0.6,
    ) -> EpisodeDomain | None:
        """Pattern completion: restore full context from partial cue.

        Returns the single best match above threshold, or None.
        """
        query_embedding = await EmbeddingService.embed_async(partial_cue)

        async with self.session_maker() as session:
            stmt = (
                select(EpisodeORM)
                .where(EpisodeORM.embedding.isnot(None))
                .order_by(EpisodeORM.embedding.cosine_distance(query_embedding))
                .limit(1)
            )
            result = await session.execute(stmt)
            episode = result.scalars().first()

        if episode is None or episode.embedding is None:
            return None

        similarity = EmbeddingService.cosine_similarity(
            query_embedding, episode.embedding
        )
        if similarity < threshold:
            return None

        return self._orm_to_domain(episode)

    async def check_pattern_separation(
        self,
        embedding: list[float],
        threshold: float = 0.92,
    ) -> list[EpisodeDomain]:
        """Find episodes very similar but with different valence.

        Used before consolidation to prevent merging distinct experiences
        that happen to look similar.
        """
        async with self.session_maker() as session:
            stmt = (
                select(EpisodeORM)
                .where(EpisodeORM.embedding.isnot(None))
                .order_by(EpisodeORM.embedding.cosine_distance(embedding))
                .limit(10)
            )
            result = await session.execute(stmt)
            candidates = list(result.scalars().all())

        separated: list[EpisodeDomain] = []
        for ep in candidates:
            if ep.embedding is None:
                continue
            sim = EmbeddingService.cosine_similarity(embedding, ep.embedding)
            if sim >= threshold:
                separated.append(self._orm_to_domain(ep))

        return separated

    async def get_unconsolidated(
        self,
        since: datetime | None = None,
        limit: int = 100,
        sources: list[str] | None = None,
    ) -> list[EpisodeDomain]:
        """Get episodes not yet processed by morning session.

        Args:
            since: Optional lower bound on timestamp.
            limit: Max episodes to return.
            sources: If set, only episodes whose source is in this list.
        """
        async with self.session_maker() as session:
            stmt = (
                select(EpisodeORM)
                .where(EpisodeORM.consolidated.is_(False))
                .order_by(EpisodeORM.timestamp.desc())
                .limit(limit)
            )
            if since is not None:
                stmt = stmt.where(EpisodeORM.timestamp >= since)
            if sources:
                stmt = stmt.where(EpisodeORM.source.in_(sources))
            result = await session.execute(stmt)
            return [self._orm_to_domain(ep) for ep in result.scalars().all()]

    async def mark_consolidated(
        self,
        episode_ids: list[int],
        result: str = "",
    ) -> None:
        """Mark episodes as consolidated after morning session."""
        if not episode_ids:
            return
        async with self.session_maker() as session:
            await session.execute(
                update(EpisodeORM)
                .where(EpisodeORM.id.in_(episode_ids))
                .values(consolidated=True, consolidation_result=result)
            )
            await session.commit()

    async def update_importance(
        self,
        episode_id: int,
        new_importance: float,
        reconsolidation_window_hours: int = 6,
    ) -> None:
        """Update importance with reconsolidation window guard.

        Importance can only be updated if the episode was last accessed
        less than reconsolidation_window_hours ago. After that, importance
        is locked.
        """
        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(hours=reconsolidation_window_hours)

        async with self.session_maker() as session:
            stmt = select(EpisodeORM).where(EpisodeORM.id == episode_id)
            result = await session.execute(stmt)
            episode = result.scalars().first()

            if episode is None:
                return

            # Allow update if never accessed or accessed within window
            if episode.last_accessed is not None and episode.last_accessed < cutoff:
                return  # Hardened — skip silently

            episode.importance = new_importance
            await session.commit()

    async def update_valence(
        self,
        episode_id: int,
        new_valence: str,
        new_confidence: float,
    ) -> None:
        """Update somatic marker based on outcome feedback."""
        async with self.session_maker() as session:
            await session.execute(
                update(EpisodeORM)
                .where(EpisodeORM.id == episode_id)
                .values(valence=new_valence, confidence=new_confidence)
            )
            await session.commit()

    async def update_enrichment(
        self,
        episode_id: int,
        result: EnrichmentResult,
    ) -> None:
        """Overwrite Sonnet-enriched fields on an existing episode.

        Called asynchronously from ChatResponseSkill after the episode is
        recorded with placeholder values. If episode_id does not exist
        (race condition with deletion), the UPDATE silently affects zero
        rows. Does not touch content, embedding, or other columns.

        Persists `feedback_strength`, `is_feedback`, and `reasoning` into
        `metadata_json['enrichment']`, preserving any pre-existing
        top-level keys (e.g. `file_path` read by `src/core/decision.py`).
        Also marks `enrichment_status="complete"` so future phases can
        filter out in-flight enrichments if needed.
        """
        enrichment_updates: dict[str, object] = {
            "feedback_strength": result.feedback_strength,
            "is_feedback": result.is_feedback,
            "reasoning": result.reasoning,
            "arousal": result.arousal,
            "self_emotion": result.self_emotion,
            "self_arousal": result.self_arousal,
        }
        async with self.session_maker() as session:
            row = await session.execute(
                select(EpisodeORM.metadata_json).where(EpisodeORM.id == episode_id)
            )
            current_meta: str | None = row.scalar_one_or_none()
            merged_metadata = _build_merged_metadata_json(
                current_meta, enrichment_updates
            )
            await session.execute(
                update(EpisodeORM)
                .where(EpisodeORM.id == episode_id)
                .values(
                    importance=result.importance,
                    valence=result.valence,
                    confidence=result.confidence,
                    intent=result.intent,
                    emotion=result.emotion,
                    enrichment_status="complete",
                    metadata_json=merged_metadata,
                )
            )
            await session.commit()

    # --- Private methods ---

    def _check_social_rate(self, user_id: int) -> bool:
        """Check if user is within social mode rate limit."""
        now = time.monotonic()
        timestamps = self._social_timestamps.get(user_id, [])

        # Clean old entries
        timestamps = [t for t in timestamps if now - t < _SOCIAL_RATE_WINDOW]

        if len(timestamps) >= _SOCIAL_RATE_LIMIT:
            self._social_timestamps[user_id] = timestamps
            return False

        timestamps.append(now)
        self._social_timestamps[user_id] = timestamps
        return True

    @staticmethod
    def _build_tier1_embed_text(
        content: str,
        source: str,
        chat_type: str,
        role: str,
        person_name: str,
        significance: str,
        domain: str,
        importance: float,
    ) -> str:
        """Build enriched text for Tier 1 embedding."""
        # Detect message type
        if "?" in content:
            msg_type = "question"
        elif any(
            w in content.lower()
            for w in ("хорошо", "плохо", "круто", "отлично", "ужасно")
        ):
            msg_type = "feedback"
        else:
            msg_type = "statement"

        return (
            f"{content} "
            f"[source:{source} mode:{chat_type} role:{role} "
            f"person:{person_name} significance:{significance} "
            f"domain:{domain} type:{msg_type} importance:{importance:.1f}]"
        )

    @staticmethod
    def _act_r_score(
        episode: EpisodeORM,
        query_embedding: list[float],
        now: datetime,
    ) -> float:
        """Hybrid ACT-R activation score.

        Score = BLA + 2.0 * cosine_sim + 1.5 * (importance - 0.5)
        BLA = ln(n/0.5) - 0.5*ln(L) + ln(1 + t_recent^(-0.5))
        """
        n = max(episode.access_count, 1)
        created = episode.timestamp
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)

        lifetime_seconds = max((now - created).total_seconds(), 1.0)

        # Base-level activation (ACT-R optimized approximation)
        bla = math.log(n / 0.5) - 0.5 * math.log(lifetime_seconds)

        # Recency boost from last access
        last = episode.last_accessed
        if last is not None:
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            t_recent = max((now - last).total_seconds(), 1.0)
            bla += math.log(1.0 + t_recent ** (-0.5))

        # Spreading activation (cosine similarity)
        cosine_sim = 0.0
        if episode.embedding is not None:
            cosine_sim = EmbeddingService.cosine_similarity(
                query_embedding, episode.embedding
            )

        # Importance bonus (centered around 0.5)
        importance_bonus = 1.5 * (episode.importance - 0.5)

        return bla + 2.0 * cosine_sim + importance_bonus
