"""Architecture audit edge-case tests.

Covers non-obvious bugs found during cross-phase integration audit:
- handle_explicit_rejection targeting auto-managed files
- _phase_prune_index line ordering (newest entries survive)
- _phase_prune_index transactional write through .pending/
- _check_valence_conflict using enriched valence (not hardcoded keywords)
- _find_similar_file embedding cache from _phase_orient
- Social mode enrichment skip
- Cross-day boundary triplet detection
- Sensitive data in correction flow
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.memory.consolidation import ConsolidationEngine


def _make_episode(
    id: int = 1,
    content: str = "test",
    user_id: int = 12345,
    importance: float = 0.5,
    valence: str = "neutral",
    embedding: list[float] | None = None,
    **kwargs: object,
) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": id,
        "content": content,
        "user_id": user_id,
        "importance": importance,
        "valence": valence,
        "embedding": embedding or [0.1] * 384,
        "timestamp": datetime.now(tz=UTC),
        "role": "user",
        "chat_type": "personal",
        "consolidated": False,
        "source": "chat",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _setup_engine(tmp_path: Path) -> tuple[ConsolidationEngine, AsyncMock, Path]:
    ws = tmp_path / "workspace"
    personality = ws / "personality"
    personality.mkdir(parents=True)
    (personality / "core.md").write_text("# Core\nI am Zhvusha.\n")
    (personality / "genes.md").write_text("# Genes\n| Gene | Value |\n")
    (personality / "MEMORY.md").write_text(
        "# Memory Index\n- [core.md](core.md) — who I am\n"
    )

    episodic = AsyncMock()
    episodic.get_unconsolidated = AsyncMock(return_value=[])
    episodic.mark_consolidated = AsyncMock()

    people = SimpleNamespace(
        record_interaction=lambda uid: None,
    )

    engine = ConsolidationEngine(episodic, ws, people)
    return engine, episodic, ws


def _patch_embed() -> object:
    return patch(
        "src.memory.consolidation.EmbeddingService.embed",
        return_value=[0.5] * 384,
    )


def _patch_cosine(val: float = 0.5) -> object:
    return patch(
        "src.memory.consolidation.EmbeddingService.cosine_similarity",
        return_value=val,
    )


# ============================================================
# 1. handle_explicit_rejection must NOT target auto-managed files
# ============================================================


async def test_rejection_skips_reinforcements_md(tmp_path: Path) -> None:
    """handle_explicit_rejection must never target reinforcements.md.

    Before the fix, reinforcements.md could be the cosine-similarity
    best match and would be corrupted with correction content.
    """
    engine, _, ws = _setup_engine(tmp_path)

    # Seed reinforcements.md with content that could match
    (ws / "personality" / "reinforcements.md").write_text(
        "# Reinforcement Patterns\n\n"
        "## Что работает (продолжать)\n"
        "- (+0.9) `отлично` ← «короткий ответ» (episode 42)\n"
    )

    with _patch_embed(), _patch_cosine(0.95):  # Very high similarity
        action = await engine.handle_explicit_rejection(
            rejected_conclusion="короткий ответ лучше",
            nikita_correction="не всегда, зависит от контекста",
        )

    # With high cosine sim, reinforcements.md would have been the best
    # match. The fix ensures it's skipped — action targets core.md instead
    # (which also matches at 0.95) or returns None if nothing else matches.
    if action is not None:
        assert action.file_path != "reinforcements.md"

    # Verify reinforcements.md was NOT modified
    reinforcements = (ws / "personality" / "reinforcements.md").read_text(
        encoding="utf-8"
    )
    assert "CORRECTED" not in reinforcements


async def test_rejection_skips_memory_md(tmp_path: Path) -> None:
    """handle_explicit_rejection must never target MEMORY.md."""
    engine, _, ws = _setup_engine(tmp_path)

    with _patch_embed(), _patch_cosine(0.95):
        action = await engine.handle_explicit_rejection(
            rejected_conclusion="Memory Index",
            nikita_correction="это не для коррекции",
        )

    if action is not None:
        assert action.file_path != "MEMORY.md"

    memory = (ws / "personality" / "MEMORY.md").read_text(encoding="utf-8")
    assert "CORRECTED" not in memory


# ============================================================
# 2. _phase_prune_index: newest entries survive, oldest drop
# ============================================================


async def test_prune_preserves_newest_entries(tmp_path: Path) -> None:
    """When MEMORY.md exceeds 200 lines, the NEWEST entries must survive.

    Old bug: `lines[:max_lines]` kept oldest entries and silently dropped
    newly added ones from _phase_consolidate.
    """
    engine, _, _ = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Write MEMORY.md near the limit
    header = "# Memory Index\n"
    old_entries = "".join(f"- old entry {i}\n" for i in range(198))
    engine.memory_index.write_text(header + old_entries)

    # Simulate creating 5 new personality files
    from src.memory.consolidation import ConsolidationAction

    actions = [
        ConsolidationAction(
            action="create",
            file_path=f"topic/new_file_{i}.md",
            content="",
            reason=f"new insight {i}",
        )
        for i in range(5)
    ]

    await engine._phase_prune_index(actions)

    pending = (engine.pending_dir / "MEMORY.md").read_text(encoding="utf-8")

    # All 5 new files must appear in the result
    for i in range(5):
        assert f"new_file_{i}" in pending, (
            f"new_file_{i} missing — newest entries were dropped"
        )

    # Total should be <= 200 lines
    assert len(pending.split("\n")) <= 200


async def test_prune_preserves_header(tmp_path: Path) -> None:
    """Header (# lines) must always survive pruning."""
    engine, _, _ = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    header = "# Memory Index\n\n"
    entries = "".join(f"- entry {i}\n" for i in range(300))
    engine.memory_index.write_text(header + entries)

    await engine._phase_prune_index([])

    pending = (engine.pending_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert pending.startswith("# Memory Index")


# ============================================================
# 3. _phase_prune_index writes through .pending/ (transactional)
# ============================================================


async def test_prune_writes_to_pending_not_direct(tmp_path: Path) -> None:
    """_phase_prune_index must write to .pending/MEMORY.md, not directly.

    Before the fix, a crash between prune and commit would leave
    MEMORY.md with entries pointing to files still in .pending/.
    """
    engine, _, _ = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    original_content = engine.memory_index.read_text(encoding="utf-8")

    from src.memory.consolidation import ConsolidationAction

    actions = [
        ConsolidationAction(
            action="create",
            file_path="new_topic.md",
            content="new",
            reason="test",
        )
    ]

    await engine._phase_prune_index(actions)

    # Original MEMORY.md must NOT be modified
    assert engine.memory_index.read_text(encoding="utf-8") == original_content

    # Pending file must exist with the new entry
    pending = engine.pending_dir / "MEMORY.md"
    assert pending.exists()
    assert "new_topic" in pending.read_text(encoding="utf-8")


async def test_prune_appends_memory_action(tmp_path: Path) -> None:
    """_phase_prune_index must append a MEMORY.md ConsolidationAction.

    This ensures _commit_pending moves the file during the transactional
    commit phase.
    """
    engine, _, _ = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    actions: list = []
    await engine._phase_prune_index(actions)

    memory_actions = [a for a in actions if a.file_path == "MEMORY.md"]
    assert len(memory_actions) == 1
    assert memory_actions[0].action == "update"


# ============================================================
# 4. _check_valence_conflict: enriched valence, no keywords
# ============================================================


async def test_valence_conflict_negative_episode_always_flags(
    tmp_path: Path,
) -> None:
    """Negative episode matched to existing file → always conflict.

    Old bug: only flagged if file content contained hardcoded Russian
    keywords like 'хорошо' or 'успешно'. Files without these keywords
    silently merged contradictory content.
    """
    engine, _, ws = _setup_engine(tmp_path)

    # File with no Russian valence keywords — just neutral factual content
    core = ws / "personality" / "core.md"
    core.write_text("# Core\nI prefer short answers in personal mode.\n")

    # Negative episode about the same topic
    ep = _make_episode(
        id=1,
        content="короткие ответы не работают, нужно подробнее",
        valence="negative",
        embedding=[0.5] * 384,
    )

    state = await engine._phase_orient()
    result = engine._check_valence_conflict(ep, core, state)

    # Must detect conflict even though file has no keyword markers
    assert result is True


async def test_valence_conflict_neutral_never_flags(tmp_path: Path) -> None:
    """Neutral episode → never a conflict regardless of file content."""
    engine, _, ws = _setup_engine(tmp_path)

    core = ws / "personality" / "core.md"
    core.write_text("# Core\nВсё хорошо и отлично.\n")

    ep = _make_episode(valence="neutral", embedding=[0.5] * 384)

    state = await engine._phase_orient()
    result = engine._check_valence_conflict(ep, core, state)

    assert result is False


async def test_valence_conflict_positive_flags_on_corrected_file(
    tmp_path: Path,
) -> None:
    """Positive episode flags only if file has prior correction markers."""
    engine, _, ws = _setup_engine(tmp_path)

    core = ws / "personality" / "core.md"
    core.write_text(
        "# Core\n<!-- CORRECTED -->\n~~old claim~~\n**Correction:** new understanding\n"
    )

    ep = _make_episode(valence="positive", embedding=[0.5] * 384)

    state = await engine._phase_orient()
    result = engine._check_valence_conflict(ep, core, state)

    # Positive + previously corrected file → re-evaluation needed
    assert result is True


async def test_valence_conflict_positive_no_markers_no_flag(
    tmp_path: Path,
) -> None:
    """Positive episode + clean file → no conflict (reinforcement, not contradiction)."""
    engine, _, ws = _setup_engine(tmp_path)

    core = ws / "personality" / "core.md"
    core.write_text("# Core\nI like being helpful.\n")

    ep = _make_episode(valence="positive", embedding=[0.5] * 384)

    state = await engine._phase_orient()
    result = engine._check_valence_conflict(ep, core, state)

    assert result is False


# ============================================================
# 5. _find_similar_file uses cached embeddings
# ============================================================


async def test_find_similar_file_uses_cached_embeddings(tmp_path: Path) -> None:
    """_find_similar_file should use pre-computed embeddings from orient.

    Verify that topic embeddings are NOT recomputed when cache is provided.
    """
    engine, _, _ = _setup_engine(tmp_path)

    embed_calls: list[str] = []

    async def tracking_embed(text: str) -> list[float]:
        embed_calls.append(text)
        return [0.5] * 384

    state = await engine._phase_orient()

    # Now call _find_similar_file with cached embeddings
    with (
        patch(
            "src.memory.consolidation.EmbeddingService.embed_async",
            side_effect=tracking_embed,
        ),
        patch(
            "src.memory.consolidation.EmbeddingService.cosine_similarity",
            return_value=0.85,
        ),
    ):
        embed_calls.clear()
        await ConsolidationEngine._find_similar_file(
            "test content",
            state.topic_to_file,
            state.file_embeddings,
        )

    # Only the content should be embedded, NOT the topic texts
    assert len(embed_calls) == 1  # Only content[:200]
    assert embed_calls[0] == "test content"[:200]


async def test_orient_populates_file_embeddings(tmp_path: Path) -> None:
    """_phase_orient must pre-compute embeddings for all non-auto-managed files."""
    engine, _, _ = _setup_engine(tmp_path)

    with _patch_embed():
        state = await engine._phase_orient()

    # core.md and genes.md should have embeddings, but NOT MEMORY.md
    assert "core.md" in state.file_embeddings
    assert "genes.md" in state.file_embeddings
    assert "MEMORY.md" not in state.file_embeddings


# ============================================================
# 6. Social mode enrichment skip
# ============================================================


async def test_social_mode_skips_enrichment(tmp_path: Path) -> None:
    """Social mode episodes should NOT trigger background enrichment.

    Social content is truncated to 100 chars with importance=0.1 —
    Sonnet analysis would be wasted tokens.
    """
    from unittest.mock import MagicMock

    from src.skills.chat_response.skill import ChatResponseSkill

    episodic = AsyncMock()
    episodic.record = AsyncMock(return_value=42)

    skill = ChatResponseSkill(episodic=episodic)

    context = MagicMock()
    context.mode = "social"
    context.user_id = 999
    context.metadata = {"chat_id": 123}
    context.bot = MagicMock()

    with (
        patch("src.skills.chat_response.skill.get_settings") as mock_settings,
        patch("src.skills.chat_response.skill.get_people_manager") as mock_pm,
        patch("src.skills.chat_response.skill.get_workspace_path"),
        patch("src.skills.chat_response.skill.ContextLoader"),
        patch.object(skill, "_generate_response", return_value="response"),
        patch.object(skill, "_background_enrich") as mock_enrich,
    ):
        settings = MagicMock()
        settings.workspace_path = str(tmp_path)
        settings.public_info_about_nikita = ""
        mock_settings.return_value = settings
        pm = MagicMock()
        pm.get_or_create_profile.return_value = None
        pm.record_interaction.return_value = False
        pm.get_profile_for_context.return_value = ""
        pm.get_significance_level.return_value = "stranger"
        pm.get_interaction_count.return_value = 1
        mock_pm.return_value = pm

        await skill.execute("hello from social", context)

    # _background_enrich should NOT be called for social mode
    mock_enrich.assert_not_called()


# ============================================================
# 7. Cross-day triplet: feedback after action is consolidated
# ============================================================


def test_triplet_detection_respects_chronological_order() -> None:
    """Triplets must be detected across messages with different timestamps.

    Ensure the internal sort-by-timestamp works even when episodes arrive
    in score-sorted order (from _phase_gather).
    """
    t0 = datetime(2026, 4, 5, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 4, 5, 10, 1, tzinfo=UTC)
    t2 = datetime(2026, 4, 5, 10, 2, tzinfo=UTC)

    episodes = [
        # Score-sorted (highest first), but timestamps are reversed
        _make_episode(
            id=3,
            role="user",
            importance=0.9,
            user_id=12345,
            timestamp=t2,
            intent="feedback",
            metadata_json='{"enrichment":{"feedback_strength":0.8}}',
        ),
        _make_episode(
            id=2,
            role="assistant",
            importance=0.7,
            user_id=12345,
            timestamp=t1,
            content="my helpful answer",
        ),
        _make_episode(
            id=1,
            role="user",
            importance=0.3,
            user_id=12345,
            timestamp=t0,
            content="question",
        ),
    ]

    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, 12345)

    # Despite reversed input order, triplet should be detected:
    # assistant(t1) → feedback(t2)
    assert len(triplets) == 1
    assert triplets[0]["action_id"] == 2
    assert triplets[0]["feedback_id"] == 3


# ============================================================
# 8. Sensitive data must NOT leak through correction flow
# ============================================================


async def test_rejection_with_sensitive_data_finds_match(tmp_path: Path) -> None:
    """Even though correction content may be sensitive, the function
    should still work — it writes to the matched file, not creating
    new sensitive content. The sensitive filter is for _phase_consolidate
    episode processing, not corrections.

    This test verifies the correction DOES apply (it's not filtered),
    documenting the intentional asymmetry.
    """
    engine, _, ws = _setup_engine(tmp_path)

    # File with salary info already in it (was incorrectly stored)
    (ws / "personality" / "work.md").write_text("# Work\nI earn a lot of money.\n")

    with _patch_embed(), _patch_cosine(0.8):
        action = await engine.handle_explicit_rejection(
            rejected_conclusion="I earn a lot of money",
            nikita_correction="Don't store salary information",
        )

    # Correction should still be applied — it's fixing bad content
    assert action is not None
    assert "CORRECTED" in action.content


# ============================================================
# 9. _phase_consolidate full integration: negative episode triggers synthesis
# ============================================================


async def test_negative_episode_triggers_synthesis_without_keywords(
    tmp_path: Path,
) -> None:
    """A negative-valence episode should trigger contradiction synthesis
    even when the matched file contains no hardcoded keyword markers.

    This is the core integration test for the _check_valence_conflict fix.
    """
    engine, _, ws = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # File with neutral content (no valence keywords)
    core = ws / "personality" / "core.md"
    core.write_text("# Core\nI give concise technical explanations.\n")

    episodes = [
        _make_episode(
            id=1,
            content="слишком сухие технические объяснения",
            valence="negative",
            importance=0.6,
            embedding=[0.5] * 384,
        ),
    ]

    state = await engine._phase_orient()

    with (
        _patch_embed(),
        _patch_cosine(0.85),  # High similarity → match to core.md
    ):
        actions, partial = await engine._phase_consolidate(episodes, state, 12345)

    # Should have triggered contradiction synthesis
    assert len(partial.contradictions_found) > 0
    # And written synthesized content, not blind merge
    synthesis_actions = [
        a
        for a in actions
        if "contradiction" in a.reason.lower() or "synthesized" in a.reason.lower()
    ]
    assert len(synthesis_actions) > 0


# ============================================================
# 10. Enrichment race: unenriched episode can't produce triplets
# ============================================================


def test_unenriched_episodes_produce_no_triplets() -> None:
    """Episodes without enrichment (intent=None, no metadata_json) must
    not produce triplets — they haven't been analyzed yet.

    This documents the low-level triplet detector contract. The morning
    gather phase is responsible for deferring pending user enrichments so this
    detector does not receive incomplete feedback episodes.
    """
    t0 = datetime(2026, 4, 5, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 4, 5, 10, 1, tzinfo=UTC)

    episodes = [
        _make_episode(
            id=1,
            role="assistant",
            user_id=12345,
            timestamp=t0,
            content="helpful answer",
        ),
        _make_episode(
            id=2,
            role="user",
            user_id=12345,
            timestamp=t1,
            content="отлично, так держать!",
            # NO intent, NO metadata_json → unenriched
        ),
    ]

    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, 12345)

    # Zero triplets because intent != "feedback" (it's None/missing)
    assert len(triplets) == 0


# ============================================================
# 11. Auto-managed files excluded from orient topic_to_file
# ============================================================


async def test_orient_excludes_reinforcements_from_topic_map(
    tmp_path: Path,
) -> None:
    """reinforcements.md must not appear in topic_to_file.

    Without this exclusion, the per-episode merge loop in _phase_consolidate
    could target reinforcements.md via _find_similar_file and clobber it
    with raw episode content (the Phase 5 data-loss bug).
    """
    engine, _, ws = _setup_engine(tmp_path)

    (ws / "personality" / "reinforcements.md").write_text(
        "# Reinforcement Patterns\nsome content here\n"
    )

    with _patch_embed():
        state = await engine._phase_orient()

    # reinforcements.md should NOT be in topic_to_file values
    assert "reinforcements.md" not in state.topic_to_file.values()
    # But it SHOULD appear in the files list (for total_files count)
    assert "reinforcements.md" in state.files


# ============================================================
# 12. Full consolidation run is transactional end-to-end
# ============================================================


async def test_full_consolidation_memory_md_via_pending(tmp_path: Path) -> None:
    """Full run: MEMORY.md must be updated via .pending/ transactional path."""
    engine, episodic, _ws = _setup_engine(tmp_path)

    episodes = [
        _make_episode(id=1, importance=0.9, content="новый навык"),
    ]
    episodic.get_unconsolidated = AsyncMock(return_value=episodes)

    with _patch_embed(), _patch_cosine(0.3):
        await engine.run_consolidation(admin_user_id=12345)

    # After full run, .pending/ should be cleaned up
    assert not engine.pending_dir.exists()

    # MEMORY.md should exist and be valid
    assert engine.memory_index.exists()
    content = engine.memory_index.read_text(encoding="utf-8")
    assert "# Memory Index" in content
