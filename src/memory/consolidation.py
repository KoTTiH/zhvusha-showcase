"""Morning session consolidation using AutoDream pattern.

Implements :class:`src.memory.protocols.ConsolidationProtocol` — public
contract for morning-session consolidation.

Phases:

1. Orient — read personality/ index, understand current state
2. Gather — retrieve unconsolidated episodes, score and filter
3. Consolidate — compress into semantic memory (personality/ files)
4. Review staging — strategist review of learning signals (promote/merge/hold/discard)
5. Prune & Index — update MEMORY.md, keep under limits

Uses transactional writes via the ``.pending/`` directory: partial
results accumulate there during phases, and are moved atomically into
``personality/`` only after all phases succeed. On any failure the
``.pending/`` directory is cleaned up by the top-level ``try/except``
in :meth:`ConsolidationEngine.run_consolidation`.

Design note — NOT structured via :class:`src.core.pipeline.PipelineRunner`
-------------------------------------------------------------------------
The flow contains conditional phases (Phase 3 runs only when
``episodes`` is non-empty, Phase 4 always runs), cross-phase state (the
``actions`` list is accumulated and later deduplicated by
``_coalesce_actions``), and an atomic ``try/except`` that cleans up
``.pending/`` on failure. Wrapping these into linear pipeline stages
would inflate the context dataclass and break the ``.pending/``
rollback semantics without buying testability that the existing
``_phase_*`` methods already have individually. Phase 5C therefore adds
the protocol (so clients depend on an interface, not the class) and
keeps the internal orchestration monolithic. If a future phase
introduces more linearity here, revisit.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.embeddings import EmbeddingService
from src.memory.protocols import ConsolidationAction, ConsolidationResult
from src.memory.staging_parser import StagingEntry, parse_staging_file

if TYPE_CHECKING:
    from src.memory.episodic import EpisodicMemory
    from src.memory.people import PeopleManager
    from src.memory.protocols import Episode

logger = structlog.get_logger()

# Hidden subdirectories under personality/ that must be skipped during any
# recursive scan of the personality tree. `.pending/` is the transactional
# staging area for consolidation writes; `.staging/` holds Phase 2 learning
# signals that are only meant for the next-turn system prompt (via
# ContextLoader.load_personality) and `/morning` staging review (Phase 4).
_SKIP_DIRS: tuple[str, ...] = (".pending", ".staging")


def _is_skipped_path(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


# Sensitive data patterns — never write to personality/ files
_SENSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\d{4,}.*(?:руб|₽|\$|кредит|долг|зарплата)",
        r"(?:болезн|диагноз|врач|лечени)",
        r"(?:никому не говори|секрет|между нами)",
        r"(?:пароль|токен|ключ|api.key|credentials)",
    ]
]

# Phase 4 — staging review constants
_STAGING_HOLD_MAX_AGE = timedelta(days=14)
_STAGING_REVIEW_MAX_ENTRIES = 300
_STAGING_FINAL_TEXT_MAX_LEN = 800
_STAGING_EXISTING_FILES_LIMIT = 100

_MD_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_MD_FENCE_CLOSE_RE = re.compile(r"```\s*$", re.MULTILINE)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_SAFE_TARGET_RE = re.compile(r"^[a-z0-9_][a-z0-9_/-]*\.md$")
_VALID_DECISIONS = {"promote_new", "merge_existing", "discard", "hold"}

# Phase 5 — reinforcement triplets. Files managed automatically by
# consolidation itself, never valid targets for:
#   - `_phase_orient.topic_to_file` (generic merge loop clobber risk)
#   - staging review existing_files (LLM can't write auto-managed)
#   - `_phase_prune_index` pointer generation (no MEMORY.md entries)
_AUTO_MANAGED_FILES: frozenset[str] = frozenset({"MEMORY.md", "reinforcements.md"})

_REINFORCEMENT_FILE = "reinforcements.md"
_REINFORCEMENT_MAX_ENTRIES = 30
_REINFORCEMENT_STRENGTH_THRESHOLD = 0.3
_REINFORCEMENT_ACTION_TEXT_LIMIT = 200
_REINFORCEMENT_FEEDBACK_TEXT_LIMIT = 100
_REINFORCEMENT_DISPLAY_ACTION_LIMIT = 80
_REINFORCEMENT_DISPLAY_FEEDBACK_LIMIT = 80
_REINFORCEMENT_JSON_RE = re.compile(
    r"<!-- reinforcement_data\s*\n(\[.*?\])\s*\n-->", re.DOTALL
)
_VALID_TRIPLET_CHAT_TYPES: frozenset[str] = frozenset({"personal", "assistant"})


def _extract_feedback_strength(ep: object) -> float:
    """Read `feedback_strength` from `episode.metadata_json['enrichment']`.

    Returns `0.0` on any error — missing/None metadata_json, corrupted JSON,
    missing enrichment sub-key, or wrong value type. Logs at debug level to
    help catch enrichment pipeline gaps without spamming warnings.
    """
    raw = getattr(ep, "metadata_json", None)
    if not raw:
        return 0.0
    try:
        meta = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.debug(
            "triplet_missing_feedback_strength",
            episode_id=getattr(ep, "id", None),
            reason="json_decode_failed",
        )
        return 0.0
    if not isinstance(meta, dict):
        return 0.0
    enrichment = meta.get("enrichment")
    if not isinstance(enrichment, dict):
        return 0.0
    value = enrichment.get("feedback_strength", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strip_markdown_fences(text: str) -> str:
    cleaned = _MD_FENCE_RE.sub("", text, count=1)
    cleaned = _MD_FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def _parse_staging_decisions(raw: str) -> list[dict[str, object]] | None:
    """Robust JSON-array extraction with two-layer fallback."""
    if not raw:
        return None
    cleaned = _strip_markdown_fences(raw)
    try:
        parsed: object = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return (
            [item for item in parsed if isinstance(item, dict)]
            if all(isinstance(i, dict) for i in parsed)
            else None
        )

    match = _JSON_ARRAY_RE.search(cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            return [item for item in parsed if isinstance(item, dict)]
    return None


_STAGING_SYSTEM_PROMPT = (
    "Ты — редактор личности Жвуши на утренней ревизии. "
    "Подходи спокойно и рефлексивно — не торопись, взвешивай каждый сигнал. "
    "Твоя задача — отфильтровать учебные сигналы за ночь и решить судьбу каждого."
)


@dataclass
class OrientResult:
    """Output of Phase 1 — current personality state snapshot."""

    files: list[str]
    total_files: int
    total_size_bytes: int
    memory_index_content: str
    core_md_content: str
    genes_md_content: str
    topic_to_file: dict[str, str]
    # Pre-computed embeddings for each file (keyed by relative path).
    # Avoids redundant re-computation in _find_similar_file: O(N) embeds
    # during orient vs O(N*M) without caching (N=files, M=episodes).
    file_embeddings: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class StagingReviewResult:
    """Output of Phase 4 — staging review pass.

    Carries the actions to commit, held blocks (to rewrite verbatim to
    learnings_pending.md after commit), per-outcome counters, and snapshot
    file sizes used to preserve tail-writes that land during review.
    """

    actions: list[ConsolidationAction] = field(default_factory=list)
    held_blocks: list[str] = field(default_factory=list)
    promoted: int = 0
    merged: int = 0
    discarded: int = 0
    held: int = 0
    stale_discarded: int = 0
    parse_failed: int = 0
    skipped_empty: bool = False
    review_failed: bool = False
    snapshot_sizes: dict[str, int] = field(default_factory=dict)


class ConsolidationEngine:
    """Morning session consolidation using AutoDream pattern."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        workspace_root: Path,
        people_manager: PeopleManager,
    ) -> None:
        self.episodic = episodic
        self.workspace = workspace_root
        self.personality_dir = workspace_root / "personality"
        self.pending_dir = self.personality_dir / ".pending"
        self.memory_index = self.personality_dir / "MEMORY.md"
        self.people = people_manager

    async def run_consolidation(
        self,
        admin_user_id: int,
    ) -> ConsolidationResult:
        """Run full consolidation (Phases 1-4 + commit + staging cleanup).

        Phase 4 (staging review) runs even when no new episodes exist —
        staging accumulates chat signals independently of episodic
        consolidation. Uses transactional writes via .pending/ directory.
        """
        # Clean up incomplete previous run
        if self.pending_dir.exists():
            logger.warning("consolidation_incomplete_previous_run_cleanup")
            shutil.rmtree(self.pending_dir)

        self.pending_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Orient
            state = await self._phase_orient()

            # Phase 2: Gather
            episodes = await self._phase_gather(admin_user_id)

            # Phase 3: Consolidate (skipped if no new episodes)
            actions: list[ConsolidationAction] = []
            partial_result = ConsolidationResult()
            if episodes:
                actions, partial_result = await self._phase_consolidate(
                    episodes, state, admin_user_id
                )

            # Phase 3.5: Staging review (always runs)
            review = await self._phase_review_staging()
            actions.extend(review.actions)

            # Phase 4: Prune & Index
            await self._phase_prune_index(actions)

            # Dedupe by file_path so commit counts are correct when both
            # Phase 3 and Phase 4 touch the same file.
            actions = self._coalesce_actions(actions)

            # Commit: move .pending/ files to personality/
            result = self._commit_pending(actions)
            result.episodes_consolidated = len(episodes)
            result.contradictions_found = partial_result.contradictions_found
            result.reinforcements = partial_result.reinforcements
            result.staging_promoted = review.promoted
            result.staging_merged = review.merged
            result.staging_discarded = review.discarded
            result.staging_held = review.held
            result.staging_stale_discarded = review.stale_discarded
            result.staging_parse_failed = review.parse_failed
            result.staging_review_failed = review.review_failed
            result.emotional_summary = partial_result.emotional_summary

            # Write emotional snapshot to log
            self._write_emotional_snapshot(partial_result.emotional_summary)

            # Post-commit staging cleanup (only if review succeeded).
            # CRITICAL: must run AFTER _commit_pending so a commit failure
            # doesn't silently drop LearningSignals that live only in
            # .staging/ files (episodic has no copy).
            if not review.review_failed and not review.skipped_empty:
                self._apply_staging_cleanup(review)

            # Rebuild the summary line now that staging counters are set.
            self._rebuild_summary(result)

            if episodes:
                episode_ids = [ep.id for ep in episodes]
                await self.episodic.mark_consolidated(
                    episode_ids, result=result.summary
                )

            logger.info(
                "consolidation_complete",
                episodes=len(episodes),
                created=len(result.files_created),
                updated=len(result.files_updated),
                staging_promoted=result.staging_promoted,
                staging_merged=result.staging_merged,
                staging_held=result.staging_held,
            )
            return result

        except Exception:
            logger.exception("consolidation_failed")
            shutil.rmtree(self.pending_dir, ignore_errors=True)
            raise

    async def _phase_orient(self) -> OrientResult:
        """Phase 1: Read current personality state."""
        files: list[str] = []
        total_size = 0

        for path in sorted(self.personality_dir.rglob("*.md")):
            if _is_skipped_path(path):
                continue
            rel = str(path.relative_to(self.personality_dir))
            files.append(rel)
            total_size += path.stat().st_size

        # Read key files
        memory_index = ""
        if self.memory_index.exists():
            memory_index = self.memory_index.read_text(encoding="utf-8")

        core_path = self.personality_dir / "core.md"
        core_content = ""
        if core_path.exists():
            core_content = core_path.read_text(encoding="utf-8")

        genes_path = self.personality_dir / "genes.md"
        genes_content = ""
        if genes_path.exists():
            genes_content = genes_path.read_text(encoding="utf-8")

        # Build topic → file map using embeddings of first 200 chars.
        # `_AUTO_MANAGED_FILES` are skipped — the per-episode merge loop in
        # `_phase_consolidate` must never target these files as a similarity
        # match. MEMORY.md is the index (auto-written by `_phase_prune_index`)
        # and reinforcements.md is auto-managed by `_write_reinforcements_pending`
        # — blindly appending episode content to either would corrupt them.
        topic_to_file: dict[str, str] = {}
        file_embeddings: dict[str, list[float]] = {}
        for rel_path in files:
            if rel_path in _AUTO_MANAGED_FILES:
                continue
            full_path = self.personality_dir / rel_path
            try:
                text = full_path.read_text(encoding="utf-8")[:200]
                if text.strip():
                    topic_to_file[text.strip()] = rel_path
                    # Pre-compute embedding once — reused by _find_similar_file
                    # for every episode instead of being recomputed O(N*M) times.
                    file_embeddings[rel_path] = await EmbeddingService.embed_async(
                        text.strip()
                    )
            except OSError:
                continue

        return OrientResult(
            files=files,
            total_files=len(files),
            total_size_bytes=total_size,
            memory_index_content=memory_index,
            core_md_content=core_content,
            genes_md_content=genes_content,
            topic_to_file=topic_to_file,
            file_embeddings=file_embeddings,
        )

    async def _phase_gather(
        self,
        admin_user_id: int,
    ) -> list[Episode]:
        """Phase 2: Get episodes worth consolidating."""
        all_episodes = await self.episodic.get_unconsolidated(limit=500)

        if not all_episodes:
            return []

        episodes: list[Episode] = []
        deferred_for_enrichment = 0
        for ep in all_episodes:
            if self._should_defer_until_enriched(ep):
                deferred_for_enrichment += 1
                continue
            episodes.append(ep)

        if deferred_for_enrichment:
            logger.info(
                "consolidation_deferred_pending_enrichment",
                count=deferred_for_enrichment,
            )

        # Score each episode
        scored: list[tuple[Episode, float]] = []
        for ep in episodes:
            score = ep.importance

            # Admin boost: always process
            if ep.user_id == admin_user_id:
                score += 0.3

            # Stranger penalty: only if importance > 0.6
            if ep.user_id != admin_user_id and ep.importance <= 0.6:
                score *= 0.5

            scored.append((ep, score))

        from src.core.config import get_settings

        top_n = max(1, get_settings().consolidation_top_n)

        # Sort by score descending, take top-N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_n]]

    @staticmethod
    def _should_defer_until_enriched(episode: Episode) -> bool:
        """Keep in-flight user turns out of morning consolidation.

        User messages in personal/assistant modes receive background enrichment.
        Consolidating them while `enrichment_status="pending"` can permanently
        lose feedback intent and reinforcement triplets once the episode is
        marked consolidated. Assistant turns are kept because they are action
        context for later enriched feedback; social turns are not enriched by
        design.
        """
        if getattr(episode, "role", None) != "user":
            return False
        if getattr(episode, "chat_type", "personal") == "social":
            return False
        return getattr(episode, "enrichment_status", "complete") == "pending"

    async def _phase_consolidate(
        self,
        episodes: list[Episode],
        current_state: OrientResult,
        admin_user_id: int,
    ) -> tuple[list[ConsolidationAction], ConsolidationResult]:
        """Phase 3: Compress episodes into semantic memory.

        Writes to .pending/ directory, not directly to personality/.
        Filters sensitive data. Detects contradictions and reinforcements.
        Checks pattern separation (valence) before merging.

        IMPORTANT: people_mentioned from Tier 2 enrichment must NEVER
        trigger PeopleManager.get_or_create_profile(). Only real Telegram
        user_ids from episodes create profiles. Enrichment names are
        search context only — stored in metadata_json, never as profiles.
        This prevents phantom profiles from LLM hallucinations.
        """
        actions: list[ConsolidationAction] = []
        partial_result = ConsolidationResult()

        # Step 1: Detect three-factor reinforcement triplets. The new
        # contract (Phase 5) uses enriched `intent="feedback"` +
        # `feedback_strength` from metadata_json; each entry is a dict with
        # keys `action_id, action_text, feedback_id, feedback_text, strength,
        # valence, ts`. The flat display string is kept for summary output.
        triplets = self._find_reinforcement_triplets(episodes, admin_user_id)
        partial_result.reinforcements = [
            f"Pattern reinforced: {str(t.get('action_text', ''))[:60]}"
            for t in triplets
        ]

        # Emotional pattern analysis (pure computation, no LLM call)
        partial_result.emotional_summary = self._analyze_emotional_patterns(episodes)

        for ep in episodes:
            # Sensitive data filter
            if self._is_sensitive(ep.content):
                logger.info("consolidation_sensitive_skipped", episode_id=ep.id)
                continue

            # Check if similar topic exists
            target = await self._find_similar_file(
                ep.content,
                current_state.topic_to_file,
                current_state.file_embeddings,
            )

            if target is not None:
                # Pattern separation check before merging:
                # if existing file content has different valence from
                # this episode, do NOT merge — store separately
                original = self.personality_dir / target
                if original.exists() and ep.embedding is not None:
                    separation = self._check_valence_conflict(
                        ep, original, current_state
                    )
                    if separation:
                        # Contradiction detected — synthesize
                        synthesized = await self._synthesize_contradiction(
                            existing_text=original.read_text(encoding="utf-8")[:200],
                            new_text=ep.content[:200],
                        )
                        partial_result.contradictions_found.append(
                            f"Episode {ep.id} vs {target}: {synthesized[:80]}"
                        )

                        # Write synthesized version instead of blind append
                        pending_path = self.pending_dir / target
                        pending_path.parent.mkdir(parents=True, exist_ok=True)
                        existing = original.read_text(encoding="utf-8")
                        new_content = (
                            existing.rstrip()
                            + f"\n\n<!-- Synthesized from episode {ep.id} "
                            f"(contradiction resolved) -->\n{synthesized}\n"
                        )
                        pending_path.write_text(new_content, encoding="utf-8")
                        actions.append(
                            ConsolidationAction(
                                action="update",
                                file_path=target,
                                content=new_content,
                                reason=f"Episode {ep.id}: contradiction synthesized",
                            )
                        )
                        continue

                # No conflict — normal merge
                pending_path = self.pending_dir / target
                pending_path.parent.mkdir(parents=True, exist_ok=True)

                existing = ""
                if original.exists():
                    existing = original.read_text(encoding="utf-8")

                new_content = (
                    existing.rstrip()
                    + f"\n\n<!-- Episode {ep.id} -->\n{ep.content[:200]}\n"
                )
                pending_path.write_text(new_content, encoding="utf-8")

                actions.append(
                    ConsolidationAction(
                        action="update",
                        file_path=target,
                        content=new_content,
                        reason=f"Episode {ep.id}: similar topic found",
                    )
                )
            elif ep.importance > 0.8:
                # Create new file for high-importance episode
                category = self._categorize_topic(ep.content)
                filename = f"{category}/episode_{ep.id}.md"
                pending_path = self.pending_dir / filename
                pending_path.parent.mkdir(parents=True, exist_ok=True)

                content = f"# Episode {ep.id}\n\n{ep.content[:500]}\n"
                pending_path.write_text(content, encoding="utf-8")

                actions.append(
                    ConsolidationAction(
                        action="create",
                        file_path=filename,
                        content=content,
                        reason=f"High importance ({ep.importance:.2f})",
                    )
                )

            # Update people profiles — ONLY from real Telegram user_ids.
            # Never from enrichment people_mentioned (prevents phantoms).
            if ep.user_id != admin_user_id:
                self.people.record_interaction(ep.user_id)

        # Phase 5: persist reinforcement triplets via helper (keeps
        # `_phase_consolidate` under the mccabe complexity limit).
        self._append_reinforcement_action(triplets, actions)

        return actions, partial_result

    def _append_reinforcement_action(
        self,
        triplets: list[dict[str, object]],
        actions: list[ConsolidationAction],
    ) -> None:
        """Write `reinforcements.md` from triplets and append the resulting
        action to the consolidate actions list.

        Runs AFTER the per-episode loop so `.pending/` already reflects all
        generic writes; `_AUTO_MANAGED_FILES` guarantees the loop never
        targeted `reinforcements.md` so there is no collision. Any failure
        in the writer is logged and swallowed — Phase 5 must never break
        morning consolidation.
        """
        try:
            action = self._write_reinforcements_pending(triplets)
        except Exception:
            logger.warning("reinforcement_write_failed", exc_info=True)
            return
        if action is not None:
            actions.append(action)

    async def _phase_prune_index(
        self,
        actions: list[ConsolidationAction],
    ) -> None:
        """Phase 4: Update MEMORY.md index.

        Writes to `.pending/MEMORY.md` (transactional) so that a crash
        between prune and commit never leaves dangling entries in MEMORY.md.
        When over the line limit, the OLDEST entries are dropped (header
        preserved) so that newly created personality files always appear.
        """
        if not self.memory_index.exists():
            return

        content = self.memory_index.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Separate header (# lines, blank lines at top) from entries.
        header: list[str] = []
        entries: list[str] = []
        for line in lines:
            if not entries and (line.startswith("#") or not line.strip()):
                header.append(line)
            else:
                entries.append(line)

        # Add pointers for new files. Auto-managed files (MEMORY.md itself,
        # reinforcements.md) are excluded — they have their own write paths
        # and don't need to appear as index entries.
        for action in actions:
            if (
                action.action == "create"
                and action.file_path not in _AUTO_MANAGED_FILES
            ):
                name = Path(action.file_path).stem.replace("_", " ")
                entry = f"- [{name}]({action.file_path}) — {action.reason}"
                entries.append(entry)

        # Enforce limits: keep header + NEWEST entries (drop oldest first).
        max_lines = 200
        max_entries = max(max_lines - len(header), 1)
        if len(entries) > max_entries:
            entries = entries[-max_entries:]

        lines = header + entries

        max_kb = 25
        text = "\n".join(lines)
        while len(text.encode("utf-8")) > max_kb * 1024 and entries:
            entries.pop(0)  # drop oldest entry on size overflow too
            lines = header + entries
            text = "\n".join(lines)

        # Write through .pending/ for transactional safety — _commit_pending
        # moves the file atomically after all phases complete.
        pending_path = self.pending_dir / "MEMORY.md"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(text, encoding="utf-8")
        actions.append(
            ConsolidationAction(
                action="update",
                file_path="MEMORY.md",
                content=text,
                reason="index pruned",
            )
        )

    async def handle_explicit_rejection(
        self,
        rejected_conclusion: str,
        nikita_correction: str,
    ) -> ConsolidationAction | None:
        """Handle when Nikita says 'no, you understood wrong'.

        Search personality/ for the wrong conclusion, mark as corrected,
        write the correct version, add diary entry.
        """
        # Search personality files for the rejected conclusion
        rejected_embedding = await EmbeddingService.embed_async(rejected_conclusion)
        best_match: Path | None = None
        best_sim = 0.0

        for path in self.personality_dir.rglob("*.md"):
            if _is_skipped_path(path):
                continue
            # Auto-managed files (MEMORY.md, reinforcements.md) must never
            # be targeted by corrections — they have their own write paths.
            if path.name in _AUTO_MANAGED_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8")[:200]
            except OSError:
                continue
            if not text.strip():
                continue
            file_embedding = await EmbeddingService.embed_async(text)
            sim = EmbeddingService.cosine_similarity(rejected_embedding, file_embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = path

        if best_match is None or best_sim < 0.5:
            return None

        # Read and update the file
        rel_path = str(best_match.relative_to(self.personality_dir))
        old_content = best_match.read_text(encoding="utf-8")
        new_content = (
            old_content + f"\n\n<!-- CORRECTED -->\n"
            f"~~{rejected_conclusion}~~\n"
            f"**Correction:** {nikita_correction}\n"
        )
        best_match.write_text(new_content, encoding="utf-8")

        # Write diary entry
        diary_dir = self.workspace / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        from datetime import UTC, datetime

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        diary_file = diary_dir / f"{today}.md"
        entry = (
            f"\n## Correction\n"
            f"Nikita corrected me: '{rejected_conclusion}' → "
            f"'{nikita_correction}'\n"
        )
        with diary_file.open("a", encoding="utf-8") as f:
            f.write(entry)

        return ConsolidationAction(
            action="update",
            file_path=rel_path,
            content=new_content,
            reason=f"Corrected by Nikita: {nikita_correction[:50]}",
        )

    @staticmethod
    def _analyze_emotional_patterns(
        episodes: list[Episode],
    ) -> str:
        """Compute emotional dynamics summary from enrichment metadata.

        Pure computation — no LLM call. Returns 2-3 line summary
        for diary, or empty string if no enrichment data available.
        """
        emotion_counts: dict[str, int] = {}
        arousal_sum = 0.0
        valence_sum = 0.0
        enriched_count = 0

        for ep in episodes:
            meta_str = getattr(ep, "metadata_json", None)
            if not meta_str:
                continue
            try:
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                enrichment = meta.get("enrichment", {})
                self_emotion = enrichment.get("self_emotion")
                if self_emotion:
                    emotion_counts[self_emotion] = (
                        emotion_counts.get(self_emotion, 0) + 1
                    )
                arousal_sum += float(enrichment.get("self_arousal", 0.5))
                valence_sum += float(enrichment.get("arousal", 0.5))
                enriched_count += 1
            except (json.JSONDecodeError, AttributeError, ValueError):
                continue

        if enriched_count == 0:
            return ""

        avg_arousal = arousal_sum / enriched_count
        avg_valence = valence_sum / enriched_count
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        parts = []
        if top_emotions:
            emotion_str = ", ".join(f"{name}×{count}" for name, count in top_emotions)
            parts.append(f"Эмоции: {emotion_str}")
        parts.append(
            f"Средний self_arousal={avg_arousal:.2f}, user_arousal={avg_valence:.2f}"
        )

        return " | ".join(parts)

    def _write_emotional_snapshot(self, summary: str) -> None:
        """Append daily emotional snapshot to personality/emotional_log.md.

        Capped at 30 entries (one per day). Older entries are pruned.
        """
        if not summary:
            return

        log_path = self.personality_dir / "emotional_log.md"
        today = datetime.now(tz=UTC).date().isoformat()

        header = "# Emotional Log\n\n"
        content = log_path.read_text(encoding="utf-8") if log_path.exists() else header

        lines = [line for line in content.splitlines() if line.startswith("- [")]
        lines.append(f"- [{today}] {summary}")

        # Keep only last 30 entries
        if len(lines) > 30:
            lines = lines[-30:]

        log_path.write_text(
            header + "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _find_reinforcement_triplets(
        episodes: list[Episode],
        admin_user_id: int,
    ) -> list[dict[str, object]]:
        """Find (assistant action → admin feedback) pairs using enriched metadata.

        Replaces the legacy hardcoded evaluative-words approach with enrichment's
        `intent` classification + `feedback_strength`. A triplet is emitted
        when:

        1. An assistant episode from admin appears (personal/assistant only —
           social mode is excluded as too noisy).
        2. It is followed chronologically by an admin user episode with
           `intent="feedback"` and `abs(feedback_strength) >= 0.3`.
        3. Neither `action.content` nor `feedback.content` trips the
           sensitive-data filter.

        Weak feedback (`abs(strength) < 0.3`) does NOT consume `last_assistant`
        — the next strong feedback may still match. Non-admin assistant
        episodes are ignored entirely. Input is sorted by timestamp
        internally since `_phase_gather` returns by score, not time.
        """
        triplets: list[dict[str, object]] = []
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)

        last_assistant: Episode | None = None
        for ep in sorted_eps:
            chat_type = getattr(ep, "chat_type", "personal")
            if chat_type not in _VALID_TRIPLET_CHAT_TYPES:
                continue

            if ep.role == "assistant" and ep.user_id == admin_user_id:
                # Track as candidate action. Sensitive content disqualifies
                # the action entirely — it cannot become part of any triplet.
                if ConsolidationEngine._is_sensitive(ep.content):
                    last_assistant = None
                else:
                    last_assistant = ep
                continue

            if (
                ep.role != "user"
                or ep.user_id != admin_user_id
                or getattr(ep, "intent", None) != "feedback"
                or last_assistant is None
            ):
                continue

            strength = _extract_feedback_strength(ep)
            if abs(strength) < _REINFORCEMENT_STRENGTH_THRESHOLD:
                # Weak — skip without consuming last_assistant so the next
                # strong feedback can still match the same action.
                continue

            if ConsolidationEngine._is_sensitive(ep.content):
                # Sensitive feedback: consume the action (avoid pairing it
                # with a subsequent feedback that might be misleading).
                last_assistant = None
                continue

            triplets.append(
                {
                    "action_id": int(last_assistant.id),
                    "action_text": last_assistant.content[
                        :_REINFORCEMENT_ACTION_TEXT_LIMIT
                    ],
                    "feedback_id": int(ep.id),
                    "feedback_text": ep.content[:_REINFORCEMENT_FEEDBACK_TEXT_LIMIT],
                    "strength": strength,
                    "valence": "positive" if strength > 0 else "negative",
                    "ts": last_assistant.timestamp.isoformat(),
                }
            )
            last_assistant = None  # One feedback per action

        return triplets

    def _check_valence_conflict(
        self,
        episode: Episode,
        existing_file: Path,
        current_state: OrientResult,  # noqa: ARG002 — kept for API compat
    ) -> bool:
        """Check if episode has opposite valence to existing file content.

        Returns True if there's a valence conflict (pattern separation needed).

        Uses the episode's enriched `valence` as the primary
        signal — not hardcoded Russian keywords. Since `_find_similar_file`
        already established high cosine similarity (>0.75), the episode
        addresses the same topic as the file. A non-neutral enriched valence
        strongly suggests the episode either reinforces or contradicts the
        file's claim.

        Negative episodes always trigger synthesis — the file presumably
        contains a neutral/positive claim that this episode disagrees with.
        Positive episodes only trigger synthesis if the file already carries
        contradiction markers from a previous synthesis pass.
        """
        if episode.valence == "neutral":
            return False

        # Negative episode about a topic the file covers → contradiction.
        # Synthesis (via strategist tier) produces a nuanced "both perspectives"
        # view instead of blind merge. False positives are acceptable because
        # synthesis is cheap and produces better content than naive append.
        if episode.valence == "negative":
            return True

        # Positive episode: only flag if the file was previously synthesized
        # from a contradiction, indicating a contentious topic that deserves
        # re-evaluation with new positive evidence.
        try:
            file_content = existing_file.read_text(encoding="utf-8")
        except OSError:
            return False

        return "<!-- CORRECTED -->" in file_content or (
            "contradiction" in file_content.lower()
        )

    async def _synthesize_contradiction(
        self,
        existing_text: str,
        new_text: str,
    ) -> str:
        """Synthesize two contradicting conclusions into one balanced statement.

        Uses the strategist tier for genuine dialectical synthesis.
        Falls back to a structured template if LLM is unavailable.
        """
        from src.personality import PERSONALITY_COMPACT

        prompt = (
            "У меня есть два противоречащих друг другу вывода из опыта:\n\n"
            f"Предыдущий вывод: {existing_text.strip()}\n"
            f"Новый опыт: {new_text.strip()}\n\n"
            "Они противоречат друг другу. Подойди к этому задумчиво и спокойно. "
            "Напиши ОДИН сбалансированный "
            "вывод (2-3 предложения), который учитывает оба опыта. "
            "Найди настоящий синтез, а не компромисс.\n"
            "Пиши от первого лица как Жвуша."
        )
        try:
            from src.llm.protocols import LLMRequest
            from src.llm.router import get_router

            router = get_router()
            response = await router.generate(
                LLMRequest(
                    prompt=prompt,
                    system=PERSONALITY_COMPACT,
                    tier="strategist",
                    reasoning_effort="xhigh",
                    caller="consolidation",
                )
            )
            return response.text
        except Exception:
            logger.warning("contradiction_synthesis_llm_failed", exc_info=True)
            return self._fallback_synthesis(existing_text, new_text)

    @staticmethod
    def _fallback_synthesis(existing_text: str, new_text: str) -> str:
        """Шаблонный fallback когда LLM недоступен."""
        return (
            f"**Сбалансированный вывод** (синтез из противоречащих опытов):\n"
            f"- Предыдущее понимание: {existing_text.strip()}\n"
            f"- Новый опыт: {new_text.strip()}\n"
            f"- Вывод: оба опыта верны в разных контекстах. "
            f"Истина зависит от конкретной ситуации.\n"
        )

    # ---- Phase 4: staging review helpers ---------------------------------

    @dataclass
    class _StagingPrep:
        """Output of the pre-strategist prep step in `_phase_review_staging`."""

        fresh: list[StagingEntry]
        initial_held: list[str]
        stale_discarded: int
        parse_failed: int
        snapshot: dict[str, int]
        skipped_empty: bool

    def _prepare_staging_batch(self, staging_dir: Path) -> _StagingPrep:
        """Parse both staging files, apply stale filter + batch cap.

        Returns a `_StagingPrep` with the fresh entries strategist should review,
        any entries already destined to be held, and the snapshot sizes
        used by `_apply_staging_cleanup` to preserve concurrent writes.
        """
        immediate_path = staging_dir / "learnings_immediate.md"
        pending_path = staging_dir / "learnings_pending.md"
        snapshot = self._snapshot_staging_sizes(staging_dir)

        imm_entries, imm_errors, imm_recoverable = parse_staging_file(immediate_path)
        pnd_entries, pnd_errors, pnd_recoverable = parse_staging_file(pending_path)
        for err in imm_errors + pnd_errors:
            logger.warning("staging_parse_error", error=err)

        all_entries = imm_entries + pnd_entries
        recoverable_held = imm_recoverable + pnd_recoverable
        parse_failed = (len(imm_errors) + len(pnd_errors)) - len(recoverable_held)
        parse_failed = max(parse_failed, 0)

        now = datetime.now(tz=UTC)
        fresh: list[StagingEntry] = []
        stale_discarded = 0
        for entry in all_entries:
            if now - entry.timestamp > _STAGING_HOLD_MAX_AGE:
                stale_discarded += 1
                logger.info(
                    "staging_stale_discarded",
                    scope=entry.scope,
                    statement=entry.statement[:60],
                    age_days=(now - entry.timestamp).days,
                )
                continue
            fresh.append(entry)

        batch_held_raw: list[str] = []
        if len(fresh) > _STAGING_REVIEW_MAX_ENTRIES:
            fresh.sort(key=lambda e: e.timestamp)
            overflow = fresh[_STAGING_REVIEW_MAX_ENTRIES:]
            batch_held_raw = [e.raw_block for e in overflow]
            fresh = fresh[:_STAGING_REVIEW_MAX_ENTRIES]
            logger.warning(
                "staging_review_batch_truncated",
                kept=len(fresh),
                held_overflow=len(batch_held_raw),
            )

        initial_held = list(recoverable_held) + list(batch_held_raw)
        skipped_empty = not fresh and not initial_held
        return ConsolidationEngine._StagingPrep(
            fresh=fresh,
            initial_held=initial_held,
            stale_discarded=stale_discarded,
            parse_failed=parse_failed,
            snapshot=snapshot,
            skipped_empty=skipped_empty,
        )

    async def _call_staging_review(
        self, fresh: list[StagingEntry]
    ) -> list[dict[str, object]] | None:
        """Call strategist tier with the review prompt and return parsed decisions.

        Returns `None` on LLM error, JSON parse failure, or decision count
        mismatch. Callers treat `None` as `review_failed=True`.
        """
        # Auto-managed files (MEMORY.md index, reinforcements.md written
        # by Phase 5) are excluded from the list the strategist sees — they're managed
        # by their own write paths and must not be direct staging-review
        # targets. Without this filter the reviewer could reasonably pick them as
        # dumping grounds and the entries would be silently discarded.
        existing_files = sorted(
            str(p.relative_to(self.personality_dir))
            for p in self.personality_dir.rglob("*.md")
            if not _is_skipped_path(p) and p.name not in _AUTO_MANAGED_FILES
        )[:_STAGING_EXISTING_FILES_LIMIT]

        prompt = self._build_staging_review_prompt(fresh, existing_files)
        try:
            from src.llm.protocols import LLMRequest
            from src.llm.router import get_router

            router = get_router()
            llm_response = await router.generate(
                LLMRequest(
                    prompt=prompt,
                    system=_STAGING_SYSTEM_PROMPT,
                    tier="strategist",
                    reasoning_effort="xhigh",
                    temperature=0.2,
                    caller="consolidation",
                )
            )
            raw = llm_response.text
        except Exception:
            logger.warning("staging_review_llm_failed", exc_info=True)
            return None

        decisions = _parse_staging_decisions(raw)
        if decisions is None or len(decisions) != len(fresh):
            logger.warning(
                "staging_review_parse_failed",
                raw_sample=raw[:300] if raw else None,
                expected=len(fresh),
                got=None if decisions is None else len(decisions),
            )
            return None
        return decisions

    async def _phase_review_staging(self) -> StagingReviewResult:
        """Phase 3.5: Review .staging/ learnings via strategist tier.

        Parses learnings_{immediate,pending}.md, drops stale entries (>14 days),
        calls the strongest configured strategist model to classify entries,
        applies promoted/merged
        writes to .pending/, returns held entries for post-commit rewrite.
        Never raises — on any failure returns a result with `review_failed=True`
        and staging files untouched.
        """
        staging_dir = self.personality_dir / ".staging"
        if not staging_dir.exists():
            return StagingReviewResult(skipped_empty=True)

        prep = self._prepare_staging_batch(staging_dir)
        if prep.skipped_empty:
            logger.info("staging_review_skipped_empty")
            return StagingReviewResult(
                skipped_empty=True,
                stale_discarded=prep.stale_discarded,
                parse_failed=prep.parse_failed,
                snapshot_sizes=prep.snapshot,
            )

        if not prep.fresh:
            # Only recoverable parse-failures / overflow to re-hold — no
            # strategist call needed.
            return StagingReviewResult(
                held_blocks=prep.initial_held,
                held=len(prep.initial_held),
                stale_discarded=prep.stale_discarded,
                parse_failed=prep.parse_failed,
                snapshot_sizes=prep.snapshot,
            )

        decisions = await self._call_staging_review(prep.fresh)
        if decisions is None:
            return StagingReviewResult(
                review_failed=True,
                stale_discarded=prep.stale_discarded,
                parse_failed=prep.parse_failed,
                snapshot_sizes=prep.snapshot,
            )

        result = StagingReviewResult(
            held_blocks=list(prep.initial_held),
            held=len(prep.initial_held),
            stale_discarded=prep.stale_discarded,
            parse_failed=prep.parse_failed,
            snapshot_sizes=prep.snapshot,
        )
        for idx, (entry, decision) in enumerate(
            zip(prep.fresh, decisions, strict=True)
        ):
            self._apply_single_decision(entry, decision, idx, result)
        return result

    def _validate_write_decision(
        self,
        decision: dict[str, object],
        entry: StagingEntry,
    ) -> tuple[str, str] | None:
        """Validate a promote_new/merge_existing decision.

        Returns `(target_file, final_text)` on success, `None` on any
        validation failure (caller increments `discarded`).
        """
        target_file_raw = decision.get("target_file")
        if not isinstance(target_file_raw, str) or not _SAFE_TARGET_RE.match(
            target_file_raw
        ):
            logger.warning("staging_decision_invalid_target", target=target_file_raw)
            return None
        if ".." in target_file_raw.split("/"):
            logger.warning("staging_decision_path_traversal", target=target_file_raw)
            return None

        final_text_raw = decision.get("final_text", "")
        if not isinstance(final_text_raw, str):
            return None
        final_text = final_text_raw[:_STAGING_FINAL_TEXT_MAX_LEN].strip()
        if not final_text:
            return None

        if self._is_sensitive(final_text) or self._is_sensitive(entry.statement):
            logger.info(
                "staging_sensitive_blocked",
                scope=entry.scope,
                target=target_file_raw,
            )
            return None

        return target_file_raw, final_text

    def _resolve_effective_decision(self, decision_type: str, target_file: str) -> str:
        """Apply collision-aware downgrade/upgrade to the raw decision.

        - `promote_new` into an existing file → `merge_existing`
        - `merge_existing` into a nonexistent file → `promote_new`
        """
        target_exists = (self.pending_dir / target_file).exists() or (
            self.personality_dir / target_file
        ).exists()
        if decision_type == "promote_new" and target_exists:
            logger.info("staging_promote_collision_downgrade", target=target_file)
            return "merge_existing"
        if decision_type == "merge_existing" and not target_exists:
            return "promote_new"
        return decision_type

    def _write_promote(
        self,
        entry: StagingEntry,
        target_file: str,
        final_text: str,
        result: StagingReviewResult,
    ) -> None:
        pending_target = self.pending_dir / target_file
        pending_target.parent.mkdir(parents=True, exist_ok=True)
        ts_label = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        content = (
            f"# {entry.scope}: {entry.statement[:50]}\n\n"
            f"{final_text}\n\n"
            f"<!-- staged {ts_label} episode {entry.episode_id} -->\n"
        )
        pending_target.write_text(content, encoding="utf-8")
        result.actions.append(
            ConsolidationAction(
                action="create",
                file_path=target_file,
                content=content,
                reason=f"staging promote: {entry.statement[:50]}",
            )
        )
        result.promoted += 1

    def _write_merge(
        self,
        entry: StagingEntry,
        target_file: str,
        final_text: str,
        result: StagingReviewResult,
    ) -> None:
        pending_target = self.pending_dir / target_file
        pending_target.parent.mkdir(parents=True, exist_ok=True)
        ts_label = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        existing = self._read_current_content(target_file)
        merged = (
            existing.rstrip()
            + f"\n\n<!-- staging review {ts_label} episode {entry.episode_id} -->\n"
            + f"{final_text}\n"
        )
        pending_target.write_text(merged, encoding="utf-8")
        result.actions.append(
            ConsolidationAction(
                action="update",
                file_path=target_file,
                content=merged,
                reason=f"staging merge: {entry.statement[:50]}",
            )
        )
        result.merged += 1

    def _apply_single_decision(
        self,
        entry: StagingEntry,
        decision: dict[str, object],
        idx: int,
        result: StagingReviewResult,
    ) -> None:
        """Apply one strategist decision to `result`. Never raises."""
        # Cross-check entry_index (the reviewer must preserve input order).
        if decision.get("entry_index") != idx:
            logger.warning(
                "staging_decision_index_mismatch",
                expected=idx,
                got=decision.get("entry_index"),
            )
            result.discarded += 1
            return

        decision_type = decision.get("decision")
        if decision_type not in _VALID_DECISIONS:
            logger.warning("staging_decision_invalid_type", decision=decision_type)
            result.discarded += 1
            return

        if decision_type == "discard":
            logger.info(
                "staging_discarded",
                scope=entry.scope,
                reason=str(decision.get("reason", ""))[:120],
            )
            result.discarded += 1
            return

        if decision_type == "hold":
            logger.info(
                "staging_held",
                scope=entry.scope,
                reason=str(decision.get("reason", ""))[:120],
            )
            result.held_blocks.append(entry.raw_block)
            result.held += 1
            return

        validated = self._validate_write_decision(decision, entry)
        if validated is None:
            result.discarded += 1
            return
        target_file, final_text = validated

        effective = self._resolve_effective_decision(str(decision_type), target_file)
        if effective == "promote_new":
            self._write_promote(entry, target_file, final_text, result)
        else:
            self._write_merge(entry, target_file, final_text, result)

    def _apply_staging_cleanup(self, review: StagingReviewResult) -> None:
        """Post-commit: rewrite staging files with held blocks + tail writes.

        Called ONLY after `_commit_pending` returns successfully. Failures
        here are logged but never raised — personality is already durable.
        """
        staging_dir = self.personality_dir / ".staging"
        immediate_path = staging_dir / "learnings_immediate.md"
        pending_path = staging_dir / "learnings_pending.md"

        try:
            immediate_tail = self._read_tail(
                immediate_path,
                review.snapshot_sizes.get("learnings_immediate.md", 0),
            )
            pending_tail = self._read_tail(
                pending_path,
                review.snapshot_sizes.get("learnings_pending.md", 0),
            )

            # Rewrite pending: held blocks (verbatim) + any tail writes.
            if review.held_blocks or pending_tail.strip():
                pending_path.parent.mkdir(parents=True, exist_ok=True)
                parts = [b for b in review.held_blocks if b]
                if pending_tail.strip():
                    parts.append(pending_tail)
                new_pending = "\n\n".join(parts).rstrip() + "\n"
                pending_path.write_text(new_pending, encoding="utf-8")
            elif pending_path.exists():
                pending_path.unlink()

            # Drain immediate: only tail writes survive.
            if immediate_tail.strip():
                immediate_path.parent.mkdir(parents=True, exist_ok=True)
                immediate_path.write_text(immediate_tail, encoding="utf-8")
            elif immediate_path.exists():
                immediate_path.unlink()

        except OSError:
            logger.warning(
                "staging_cleanup_failed",
                held_count=len(review.held_blocks),
                exc_info=True,
            )

    def _read_current_content(self, rel_path: str) -> str:
        """Read a personality file preferring `.pending/` (Phase 3 writes)
        over `personality/` (committed state). Returns empty string if
        neither exists."""
        pending = self.pending_dir / rel_path
        if pending.exists():
            try:
                return pending.read_text(encoding="utf-8")
            except OSError:
                return ""
        final = self.personality_dir / rel_path
        if final.exists():
            try:
                return final.read_text(encoding="utf-8")
            except OSError:
                return ""
        return ""

    @staticmethod
    def _snapshot_staging_sizes(staging_dir: Path) -> dict[str, int]:
        """Record current byte sizes of both staging files."""
        sizes: dict[str, int] = {}
        for name in ("learnings_immediate.md", "learnings_pending.md"):
            path = staging_dir / name
            try:
                sizes[name] = path.stat().st_size if path.exists() else 0
            except OSError:
                sizes[name] = 0
        return sizes

    @staticmethod
    def _read_tail(path: Path, from_byte: int) -> str:
        """Read bytes from `from_byte` to EOF as UTF-8, ignoring errors."""
        if not path.exists():
            return ""
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        if size <= from_byte:
            return ""
        try:
            with path.open("rb") as f:
                f.seek(from_byte)
                return f.read().decode("utf-8", errors="ignore")
        except OSError:
            return ""

    @staticmethod
    def _coalesce_actions(
        actions: list[ConsolidationAction],
    ) -> list[ConsolidationAction]:
        """Dedupe actions by `file_path`, last-writer wins.

        Multiple actions may target the same file when Phase 3 and Phase 4
        both write to e.g. `core.md` — the disk content already reflects all
        accumulated writes, so `_commit_pending` only needs one move per
        file. The last action per path wins because it was written last.
        """
        by_path: dict[str, ConsolidationAction] = {}
        for action in actions:
            by_path[action.file_path] = action
        return list(by_path.values())

    @staticmethod
    def _build_staging_review_prompt(
        entries: list[StagingEntry],
        existing_files: list[str],
    ) -> str:
        """Render the strategist review prompt in Russian with XML-tagged sections."""
        entry_lines: list[str] = []
        for idx, entry in enumerate(entries):
            ts_label = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            entry_lines.append(f"### Entry {idx}")
            entry_lines.append(f"- type: {entry.type}")
            entry_lines.append(f"- scope: {entry.scope}")
            entry_lines.append(f"- statement: {entry.statement}")
            entry_lines.append(f"- confidence: {entry.confidence}")
            entry_lines.append(f"- timestamp: {ts_label}")
            entry_lines.append(f"- source: {entry.source_file}")
            if entry.original_claim:
                entry_lines.append(f"- original_claim: {entry.original_claim}")
            entry_lines.append("")
        staging_block = "\n".join(entry_lines)

        files_block = "\n".join(f"- {p}" for p in existing_files) or "(пусто)"

        schema = (
            "[\n"
            "  {\n"
            '    "entry_index": <int, совпадает с номером Entry>,\n'
            '    "decision": "promote_new" | "merge_existing" | "discard" | "hold",\n'
            '    "target_file": "<путь относительно personality/, *.md>" | null,\n'
            '    "final_text": "<текст для записи, 1-800 символов>",\n'
            '    "reason": "<короткое объяснение>"\n'
            "  }\n"
            "]"
        )

        examples = (
            "ПРИМЕРЫ:\n"
            "- Сильное правило, которого ещё нет в personality/ → promote_new с "
            "target_file типа values/rule_name.md\n"
            "- Дополнение к существующему правилу → merge_existing в тот же файл\n"
            "- Разовое настроение ('сегодня лень') → discard\n"
            "- Возможно правило, но confidence 0.5-0.8, нужно больше данных → hold"
        )

        return (
            "<CONTEXT>\n"
            "Жвуша накопила учебные сигналы из разговоров за ночь. Тебе нужно "
            "отсортировать каждый и решить, что с ним делать: добавить в "
            "постоянную личность, merge'нуть в существующий файл, отбросить "
            "как шум, или отложить до подтверждения.\n"
            "</CONTEXT>\n\n"
            f"<STAGING>\n{staging_block}\n</STAGING>\n\n"
            f"<EXISTING_FILES>\n{files_block}\n</EXISTING_FILES>\n\n"
            f"<SCHEMA>\n{schema}\n</SCHEMA>\n\n"
            f"<EXAMPLES>\n{examples}\n</EXAMPLES>\n\n"
            "<INSTRUCTION>\n"
            "Ответь JSON-массивом из ровно N элементов, по одному на каждую "
            f"Entry выше (N = {len(entries)}), в том же порядке. "
            "entry_index должен совпадать с номером Entry. Никакого "
            "markdown-обрамления, только массив.\n"
            "</INSTRUCTION>"
        )

    # ---- end Phase 4 helpers ---------------------------------------------

    def _commit_pending(
        self, actions: list[ConsolidationAction]
    ) -> ConsolidationResult:
        """Move files from .pending/ to personality/.

        The summary string is built later by `_rebuild_summary` once staging
        counters are merged into the result. This method only handles the
        move operation and pending-dir cleanup.
        """
        result = ConsolidationResult()

        for action in actions:
            pending_path = self.pending_dir / action.file_path
            target_path = self.personality_dir / action.file_path

            if not pending_path.exists():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pending_path), str(target_path))

            if action.action == "create":
                result.files_created.append(action.file_path)
            elif action.action == "update":
                result.files_updated.append(action.file_path)

        # Clean up pending directory
        shutil.rmtree(self.pending_dir, ignore_errors=True)

        return result

    @staticmethod
    def _rebuild_summary(result: ConsolidationResult) -> None:
        """Build `result.summary` from both consolidation and staging counters.

        Mutates `result` in place. Called after staging counters have been
        merged into the result object in `run_consolidation`.
        """
        parts: list[str] = []
        if result.files_created:
            parts.append(f"Created {len(result.files_created)} files")
        if result.files_updated:
            parts.append(f"Updated {len(result.files_updated)} files")
        if result.contradictions_found:
            parts.append(f"Resolved {len(result.contradictions_found)} contradictions")
        if result.reinforcements:
            parts.append(f"Found {len(result.reinforcements)} reinforcements")
        if result.staging_promoted or result.staging_merged:
            parts.append(
                f"Staging: +{result.staging_promoted} promoted, "
                f"~{result.staging_merged} merged, "
                f"-{result.staging_discarded} discarded, "
                f"↻{result.staging_held} held"
            )
        if result.staging_stale_discarded:
            parts.append(f"Stale discarded: {result.staging_stale_discarded}")
        if result.staging_review_failed:
            parts.append("Staging review failed (skipped)")
        result.summary = ". ".join(parts) or "No changes."

    @staticmethod
    def _is_sensitive(content: str) -> bool:
        """Check if content contains sensitive data."""
        return any(p.search(content) for p in _SENSITIVE_PATTERNS)

    # --- Phase 5: reinforcement triplet persistence ---

    @staticmethod
    def _parse_reinforcements_file(text: str) -> list[dict[str, object]]:
        """Extract the triplet list from the JSON blob embedded in a
        `reinforcements.md` file.

        Returns `[]` if no blob, invalid JSON, or non-list top-level type.
        Filters out any non-dict entries defensively.
        """
        if not text:
            return []
        match = _REINFORCEMENT_JSON_RE.search(text)
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):  # pragma: no cover — regex enforces list shape
            return []
        return [d for d in data if isinstance(d, dict)]

    @staticmethod
    def _triplet_strength(triplet: dict[str, object]) -> float:
        """Safely coerce the `strength` field of a triplet dict to float.

        Returns `0.0` for missing, None, or unconvertible values.
        """
        raw = triplet.get("strength", 0.0)
        if isinstance(raw, int | float):
            return float(raw)
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_triplet_line(triplet: dict[str, object]) -> str:
        strength = ConsolidationEngine._triplet_strength(triplet)
        sign = "+" if strength >= 0 else ""
        # Escape `-->` in human-readable bullets so only the final JSON blob
        # closing marker appears as a literal `-->`. Round-trip accuracy is
        # preserved via the unescaped JSON blob at the end of the file.
        action = str(triplet.get("action_text", ""))[
            :_REINFORCEMENT_DISPLAY_ACTION_LIMIT
        ].replace("-->", "-- >")
        feedback = str(triplet.get("feedback_text", ""))[
            :_REINFORCEMENT_DISPLAY_FEEDBACK_LIMIT
        ].replace("-->", "-- >")
        eid = triplet.get("action_id", "?")
        return f"- ({sign}{strength:.1f}) `{feedback}` ← «{action}» (episode {eid})"

    @staticmethod
    def _render_reinforcements_file(triplets: list[dict[str, object]]) -> str:
        """Render triplet list as human markdown + JSON round-trip blob.

        The blob is placed at the end of the file (attention-tail position)
        and escapes any literal `-->` in string fields so the HTML comment
        terminator stays unique.
        """
        positive = sorted(
            [t for t in triplets if ConsolidationEngine._triplet_strength(t) > 0],
            key=ConsolidationEngine._triplet_strength,
            reverse=True,
        )
        negative = sorted(
            [t for t in triplets if ConsolidationEngine._triplet_strength(t) < 0],
            key=ConsolidationEngine._triplet_strength,
        )

        lines = [
            "# Reinforcement Patterns",
            "",
            "Автогенерируется в /morning из feedback-эпизодов. "
            "Не редактировать вручную.",
            "",
            "## Что работает (продолжать)",
            "",
        ]
        for t in positive:
            lines.append(ConsolidationEngine._format_triplet_line(t))
        lines.append("")
        lines.append("## Что не работает (избегать)")
        lines.append("")
        for t in negative:
            lines.append(ConsolidationEngine._format_triplet_line(t))
        lines.append("")

        # JSON round-trip blob. Escape `-->` so the closing comment marker
        # stays unique (paranoia — very unlikely in Russian chat content).
        raw_json = json.dumps(triplets, ensure_ascii=False)
        safe_json = raw_json.replace("-->", "--\\u003e")
        lines.append("<!-- reinforcement_data")
        lines.append(safe_json)
        lines.append("-->")
        return "\n".join(lines)

    def _write_reinforcements_pending(
        self,
        new_triplets: list[dict[str, object]],
    ) -> ConsolidationAction | None:
        """Merge new triplets with existing `reinforcements.md`, write to
        `.pending/`.

        Returns `None` if nothing to persist (no triplets, or all filtered
        as sensitive). Otherwise returns a `ConsolidationAction` for the
        caller to append to the `actions` list so `_commit_pending` moves
        the file during the transactional commit phase.

        Semantics:
        - Dedup by `(action_id, feedback_id)` across existing + new.
        - Sensitive-content filter (belt-and-suspenders) on both
          `action_text` and `feedback_text`.
        - Cap at `_REINFORCEMENT_MAX_ENTRIES` by `abs(strength)`.
        - Reads existing via `_read_current_content` so any earlier Phase 3
          write to `.pending/reinforcements.md` is preserved.
        """
        if not new_triplets:
            return None

        safe = [
            t
            for t in new_triplets
            if not self._is_sensitive(str(t.get("action_text", "")))
            and not self._is_sensitive(str(t.get("feedback_text", "")))
        ]
        if not safe:
            return None

        existing_text = self._read_current_content(_REINFORCEMENT_FILE)
        existing = self._parse_reinforcements_file(existing_text)

        seen_keys: set[tuple[object, object]] = {
            (d.get("action_id"), d.get("feedback_id")) for d in existing
        }
        merged: list[dict[str, object]] = list(existing)
        for t in safe:
            key = (t.get("action_id"), t.get("feedback_id"))
            if key not in seen_keys:
                merged.append(t)
                seen_keys.add(key)

        merged.sort(
            key=lambda t: abs(ConsolidationEngine._triplet_strength(t)),
            reverse=True,
        )
        merged = merged[:_REINFORCEMENT_MAX_ENTRIES]

        if not merged:
            return None

        rendered = self._render_reinforcements_file(merged)
        pending_path = self.pending_dir / _REINFORCEMENT_FILE
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(rendered, encoding="utf-8")

        file_existed = (self.personality_dir / _REINFORCEMENT_FILE).exists()
        pos_count = sum(
            1 for t in merged if ConsolidationEngine._triplet_strength(t) > 0
        )
        neg_count = sum(
            1 for t in merged if ConsolidationEngine._triplet_strength(t) < 0
        )
        return ConsolidationAction(
            action="update" if file_existed else "create",
            file_path=_REINFORCEMENT_FILE,
            content=rendered,
            reason=f"reinforcement patterns: +{pos_count}/-{neg_count}",
        )

    @staticmethod
    async def _find_similar_file(
        content: str,
        topic_to_file: dict[str, str],
        file_embeddings: dict[str, list[float]] | None = None,
    ) -> str | None:
        """Find the most similar existing personality file.

        Uses pre-computed `file_embeddings` from `_phase_orient` when
        available, falling back to on-the-fly embedding for backward
        compatibility. This reduces embedding calls from O(N*M) to O(M)
        where N=files, M=episodes.
        """
        if not topic_to_file:
            return None

        content_embedding = await EmbeddingService.embed_async(content[:200])
        best_file: str | None = None
        best_sim = 0.0

        for topic_text, file_path in topic_to_file.items():
            if file_embeddings and file_path in file_embeddings:
                topic_embedding = file_embeddings[file_path]
            else:
                topic_embedding = await EmbeddingService.embed_async(topic_text)
            sim = EmbeddingService.cosine_similarity(content_embedding, topic_embedding)
            if sim > best_sim:
                best_sim = sim
                best_file = file_path

        if best_sim > 0.75:
            return best_file
        return None

    @staticmethod
    def _categorize_topic(content: str) -> str:
        """Classify topic into personality subdirectory."""
        lower = content.lower()
        if any(w in lower for w in ("навык", "умен", "техник", "инструмент")):
            return "skills"
        if any(w in lower for w in ("принцип", "правил", "ценност", "важно")):
            return "values"
        if any(w in lower for w in ("понял", "заметил", "ошибл", "осознал")):
            return "meta"
        return "insights"
