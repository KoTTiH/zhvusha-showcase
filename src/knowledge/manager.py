"""Zhvusha's knowledge base manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import structlog

from src.embeddings import EmbeddingService

if TYPE_CHECKING:
    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()


@dataclass
class KnowledgeEntry:
    path: str
    category: str
    topic: str
    content: str
    last_updated: datetime
    source: str


class KnowledgeManager:
    """Manages Zhvusha's knowledge base.

    knowledge/ is Zhvusha's long-term factual memory.
    Different from personality/ (which is about WHO she is).
    knowledge/ is about WHAT she knows.
    """

    def __init__(
        self,
        workspace_root: Path,
        episodic: EpisodicMemory | None = None,
    ) -> None:
        self._knowledge_dir = workspace_root / "knowledge"
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._episodic = episodic

    async def save_knowledge(
        self,
        topic: str,
        content: str,
        source: str,
        category: str,
    ) -> Path:
        """Save a knowledge entry. Merges if similar topic exists."""
        category_dir = self._knowledge_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing similar topic
        existing = await self._find_similar(topic, category)
        if existing is not None:
            merged = await self._merge_via_llm(
                existing.read_text(encoding="utf-8"), content
            )
            existing.write_text(f"# {topic}\n\n{merged}\n", encoding="utf-8")
            file_path = existing
            logger.info("knowledge_updated", path=str(file_path), topic=topic)
        else:
            slug = self._slugify(topic)
            file_path = category_dir / f"{slug}.md"
            file_path.write_text(f"# {topic}\n\n{content}\n", encoding="utf-8")
            logger.info("knowledge_created", path=str(file_path), topic=topic)

        # Record episode for searchability
        if self._episodic is not None:
            rel_path = str(file_path.relative_to(self._knowledge_dir))
            await self._episodic.record(
                content=f"Knowledge: {topic} — {content[:200]}",
                user_id=0,
                chat_type="personal",
                role="assistant",
                source="knowledge",
                importance=0.6,
                metadata={
                    "file_path": rel_path,
                    "category": category,
                    "original_source": source,
                },
            )

        return file_path

    async def _find_similar(self, topic: str, category: str) -> Path | None:
        """Find existing file with similar topic (cosine > 0.75)."""
        category_dir = self._knowledge_dir / category
        if not category_dir.exists():
            return None

        topic_embedding = await EmbeddingService.embed_async(topic)

        best_path: Path | None = None
        best_sim = 0.0

        for md_file in category_dir.glob("*.md"):
            # Use first line (title) for comparison
            first_line = md_file.read_text(encoding="utf-8").split("\n")[0]
            title = first_line.lstrip("# ").strip()
            if not title:
                continue

            file_embedding = await EmbeddingService.embed_async(title)
            sim = EmbeddingService.cosine_similarity(topic_embedding, file_embedding)

            if sim > best_sim:
                best_sim = sim
                best_path = md_file

        if best_sim > 0.75 and best_path is not None:
            return best_path
        return None

    async def _merge_via_llm(self, old_content: str, new_content: str) -> str:
        """Merge old and new content via LLM (strategist — morning session)."""
        try:
            from src.llm.protocols import LLMRequest
            from src.llm.router import get_router

            router = get_router()
            response = await router.generate(
                LLMRequest(
                    prompt=(
                        f"Объедини старую и новую информацию в один связный документ.\n"
                        f"Подходи аналитически: убери дубликаты, сохрани самое актуальное.\n"
                        f"Не добавляй оценочных суждений — только факты и связи.\n\n"
                        f"Старое:\n{old_content[:1500]}\n\n"
                        f"Новое:\n{new_content[:1500]}"
                    ),
                    tier="strategist",
                    caller="knowledge_merge",
                )
            )
            return response.text
        except Exception:
            logger.warning("knowledge_merge_failed", exc_info=True)
            return f"{old_content}\n\n---\n\n{new_content}"

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[KnowledgeEntry]:
        """Search across all knowledge files by topic/content match."""
        if not self._knowledge_dir.exists():
            return []

        entries: list[tuple[float, KnowledgeEntry]] = []
        query_embedding = await EmbeddingService.embed_async(query)

        for md_file in self._knowledge_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            first_line = content.split("\n")[0]
            topic = first_line.lstrip("# ").strip()

            # Score by embedding similarity of topic
            file_embedding = await EmbeddingService.embed_async(
                topic if topic else content[:200]
            )
            sim = EmbeddingService.cosine_similarity(query_embedding, file_embedding)

            # Also check simple keyword match
            if query.lower() in content.lower():
                sim = max(sim, 0.6)

            category = md_file.parent.name
            rel_path = str(md_file.relative_to(self._knowledge_dir))

            stat = md_file.stat()
            last_updated = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

            entries.append(
                (
                    sim,
                    KnowledgeEntry(
                        path=rel_path,
                        category=category,
                        topic=topic,
                        content=content,
                        last_updated=last_updated,
                        source="",
                    ),
                )
            )

        # Sort by similarity, return top N
        entries.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in entries[:limit]]

    async def get_relevant_for_context(
        self,
        current_topics: list[str],
        limit: int = 3,
    ) -> str:
        """Get knowledge relevant to current conversation.

        Returns concatenated relevant knowledge, max ~500 tokens total.
        """
        all_results: list[KnowledgeEntry] = []
        seen_paths: set[str] = set()

        for topic in current_topics:
            results = await self.search(topic, limit=limit)
            for r in results:
                if r.path not in seen_paths:
                    seen_paths.add(r.path)
                    all_results.append(r)

        # Build context string, respecting approximate token limit
        context_parts: list[str] = []
        total_chars = 0
        max_chars = 4000  # ~1000 tokens

        for entry in all_results[:limit]:
            snippet = entry.content[:800]
            if total_chars + len(snippet) > max_chars:
                break
            context_parts.append(f"[{entry.category}/{entry.topic}]\n{snippet}")
            total_chars += len(snippet)

        return "\n\n---\n\n".join(context_parts)

    async def cleanup_stale(
        self,
        max_age_days: int = 90,
    ) -> list[str]:
        """Remove knowledge entries not accessed in N days."""
        if not self._knowledge_dir.exists():
            return []

        cutoff = datetime.now(tz=UTC) - timedelta(days=max_age_days)
        removed: list[str] = []

        for md_file in self._knowledge_dir.rglob("*.md"):
            stat = md_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

            if mtime < cutoff:
                rel_path = str(md_file.relative_to(self._knowledge_dir))
                md_file.unlink()
                removed.append(rel_path)
                logger.info("knowledge_stale_removed", path=rel_path)

        return removed

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filename-safe slug."""
        slug = text.lower().strip()
        # Replace spaces and special chars
        for char in ' /\\:*?"<>|':
            slug = slug.replace(char, "_")
        # Collapse multiple underscores
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug[:80].rstrip("_")
