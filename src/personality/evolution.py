"""Personality tree evolution — growing, maturing, annotating."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.embeddings import EmbeddingService
from src.personality.protocols import PersonalityEvolutionProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory import Episode

logger = structlog.get_logger()


class PersonalityEvolution(PersonalityEvolutionProtocol):
    """Manages the growing personality tree.

    Decides when to create new files, when to update existing,
    when to snapshot and rewrite core.md.
    """

    def __init__(self, personality_dir: Path) -> None:
        self.root = personality_dir
        self.history = personality_dir / "history" / "snapshots"

    async def should_create_new_file(
        self,
        topic: str,
        mention_count: int,
        max_importance: float,
    ) -> bool:
        """Check if a new personality file should be created.

        Returns True if topic mentioned 3+ times OR importance > 0.8,
        AND no existing file covers this topic.
        """
        if mention_count < 3 and max_importance <= 0.8:
            return False

        target = await self.get_target_file(topic)
        return target is None

    async def get_target_file(self, topic: str) -> Path | None:
        """Find the right file for a topic.

        Embeds topic and each file's first 200 chars, returns highest
        cosine similarity if > 0.75, else None.
        """
        topic_embedding = await EmbeddingService.embed_async(topic)
        best_path: Path | None = None
        best_sim = 0.0

        for path in sorted(self.root.rglob("*.md")):
            if (
                ".pending" in path.parts
                or ".staging" in path.parts
                or path.name.startswith(".")
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")[:200]
            except OSError:
                continue
            if not text.strip():
                continue

            file_embedding = await EmbeddingService.embed_async(text)
            sim = EmbeddingService.cosine_similarity(topic_embedding, file_embedding)
            if sim > best_sim:
                best_sim = sim
                best_path = path

        if best_sim > 0.75:
            return best_path
        return None

    async def evolve_core(
        self,
        new_insights: list[str],
        max_lines: int = 30,
    ) -> bool:
        """Update core.md with new insights.

        Skips duplicates (cosine > 0.85 with existing lines).
        Snapshots when exceeding max_lines.
        Returns True if core.md was updated.
        """
        core_path = self.root / "core.md"
        if not core_path.exists():
            return False

        content = core_path.read_text(encoding="utf-8")
        existing_lines = [ln for ln in content.split("\n") if ln.strip()]

        # Pre-compute embeddings for existing lines (O(N) instead of O(NxM))
        existing_embeddings: list[list[float]] = []
        for line in existing_lines:
            existing_embeddings.append(await EmbeddingService.embed_async(line))

        # Filter out duplicate insights
        new_unique: list[str] = []
        for insight in new_insights:
            insight_embedding = await EmbeddingService.embed_async(insight)
            is_duplicate = any(
                EmbeddingService.cosine_similarity(insight_embedding, emb) > 0.85
                for emb in existing_embeddings
            )
            if not is_duplicate:
                new_unique.append(insight)

        if not new_unique:
            return False

        # Add insights
        lines = content.split("\n")
        for insight in new_unique:
            lines.append(insight)

        # Check if exceeding limit → snapshot
        non_empty = [ln for ln in lines if ln.strip()]
        if len(non_empty) > max_lines:
            self._snapshot_core(core_path)
            # Truncate to max_lines (keep header + newest insights)
            lines = lines[:5] + lines[-(max_lines - 5) :]

        core_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("core_evolved", new_insights=len(new_unique))
        return True

    async def evolve_genes(
        self,
        experience_updates: list[dict[str, str]],
    ) -> None:
        """Add dated experience annotations to genes.md.

        Does NOT change parameter values. Only adds annotations.
        """
        genes_path = self.root / "genes.md"
        if not genes_path.exists():
            return

        content = genes_path.read_text(encoding="utf-8")
        from datetime import UTC, datetime

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")

        for update in experience_updates:
            gene = update.get("gene", "")
            learned = update.get("learned", "")
            if gene and learned:
                annotation = f"\n# Learned ({gene}): {learned} (updated {today})"
                content += annotation

        genes_path.write_text(content, encoding="utf-8")
        logger.info("genes_evolved", updates=len(experience_updates))

    async def suggest_new_dimension(
        self,
        topic: str,
        episodes: list[Episode],
    ) -> Path | None:
        """Suggest where to create a new personality file.

        Keyword classification:
        - person → relationships/
        - skill/tool → skills/
        - value/principle → values/
        - self-knowledge → meta/
        """
        category = self._classify_category(topic)
        # Generate filename from topic (simplified)
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in topic[:30])
        safe_name = safe_name.strip("_").lower() or "unnamed"

        return self.root / category / f"{safe_name}.md"

    def get_personality_tree_summary(self) -> str:
        """Returns compressed personality for LLM context.

        Always loads: MEMORY.md + core.md + genes.md
        Total should stay under ~2000 tokens.
        """
        parts: list[str] = []

        memory_index = self.root / "MEMORY.md"
        if memory_index.exists():
            parts.append(memory_index.read_text(encoding="utf-8"))

        core = self.root / "core.md"
        if core.exists():
            parts.append(core.read_text(encoding="utf-8"))

        genes = self.root / "genes.md"
        if genes.exists():
            parts.append(genes.read_text(encoding="utf-8"))

        text = "\n\n---\n\n".join(parts)

        # Rough token estimate: 1 token ≈ 4 chars
        max_chars = 2000 * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (обрезано)"

        return text

    def _snapshot_core(self, core_path: Path) -> None:
        """Save current core.md to history/snapshots/."""
        self.history.mkdir(parents=True, exist_ok=True)
        existing = list(self.history.glob("core_v*.md"))
        version = len(existing) + 1
        snapshot_path = self.history / f"core_v{version}.md"
        snapshot_path.write_text(
            core_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        logger.info("core_snapshot", version=version)

    @staticmethod
    def _classify_category(content: str) -> str:
        """Classify topic into personality subdirectory."""
        lower = content.lower()
        person_words = ("имя", "человек", "друг", "знакомый", "коллег")
        skill_words = ("навык", "умен", "техник", "инструмент", "библиотек")
        value_words = ("принцип", "правил", "ценност", "важно", "убежден")
        meta_words = ("понял", "заметил", "ошибл", "осознал", "рефлекс")

        if any(w in lower for w in person_words):
            return "relationships"
        if any(w in lower for w in skill_words):
            return "skills"
        if any(w in lower for w in value_words):
            return "values"
        if any(w in lower for w in meta_words):
            return "meta"
        return "insights"
