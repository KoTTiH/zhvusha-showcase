"""Tests for ConsolidationEngine."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from src.memory.consolidation import ConsolidationEngine


def _make_episode(
    id=1,
    content="test",
    user_id=12345,
    importance=0.5,
    valence="neutral",
    embedding=None,
    **kwargs,
):
    defaults = {
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
        "enrichment_status": "complete",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _setup_engine(tmp_path: Path):
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


def _patch_embed():
    return patch(
        "src.memory.consolidation.EmbeddingService.embed",
        return_value=[0.5] * 384,
    )


def _patch_cosine(val=0.5):
    return patch(
        "src.memory.consolidation.EmbeddingService.cosine_similarity",
        return_value=val,
    )


# --- Phase 1: Orient ---


async def test_orient_reads_personality_tree(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    state = await engine._phase_orient()
    assert state.total_files >= 3  # core.md, genes.md, MEMORY.md
    assert "core.md" in state.files
    assert "genes.md" in state.files
    assert "I am Zhvusha" in state.core_md_content


async def test_orient_skips_staging_directory(tmp_path):
    """Files under personality/.staging/ must never be treated as part of
    the personality tree. Phase 2 stages learnings there via StagingWriter
    and only ContextLoader.load_personality() reads them — consolidation
    must NOT pull them into core.md or count them as files."""
    engine, _, ws = _setup_engine(tmp_path)
    staging_dir = ws / "personality" / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "learnings_immediate.md").write_text(
        "## [rule] tone\n**Statement:** staging must not leak\n",
        encoding="utf-8",
    )
    (staging_dir / "learnings_pending.md").write_text(
        "## [preference] preferences\nshould not be here\n",
        encoding="utf-8",
    )

    state = await engine._phase_orient()

    staging_files = [f for f in state.files if ".staging" in f]
    assert staging_files == [], (
        f"staging files must be filtered out of personality tree, got: {staging_files}"
    )
    assert "staging must not leak" not in state.core_md_content


async def test_orient_counts_total_size(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    state = await engine._phase_orient()
    assert state.total_size_bytes > 0


# --- Phase 2: Gather ---


async def test_gather_prioritizes_admin(tmp_path):
    engine, episodic, _ = _setup_engine(tmp_path)
    admin_id = 12345

    episodes = [
        _make_episode(id=1, user_id=admin_id, importance=0.3),
        _make_episode(id=2, user_id=999, importance=0.9),
    ]
    episodic.get_unconsolidated = AsyncMock(return_value=episodes)

    result = await engine._phase_gather(admin_id)
    # Admin gets boost (+0.3), so admin episode scores 0.6 (0.3+0.3)
    # Stranger with importance 0.9 scores 0.9
    assert len(result) == 2


async def test_gather_applies_stranger_penalty(tmp_path):
    engine, episodic, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(id=1, user_id=999, importance=0.4),  # Below 0.6 → penalty
    ]
    episodic.get_unconsolidated = AsyncMock(return_value=episodes)

    result = await engine._phase_gather(admin_user_id=12345)
    assert len(result) == 1  # Still included, but with lower score


async def test_gather_defers_pending_user_enrichment_but_keeps_action_context(
    tmp_path,
):
    engine, episodic, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(
            id=1,
            role="assistant",
            importance=0.7,
            enrichment_status="pending",
        ),
        _make_episode(
            id=2,
            role="user",
            importance=0.9,
            enrichment_status="pending",
        ),
        _make_episode(
            id=3,
            role="user",
            importance=0.8,
            enrichment_status="complete",
        ),
    ]
    episodic.get_unconsolidated = AsyncMock(return_value=episodes)

    result = await engine._phase_gather(admin_user_id=12345)

    assert [ep.id for ep in result] == [3, 1]


async def test_run_consolidation_does_not_mark_pending_user_enrichment(
    tmp_path,
):
    engine, episodic, _ = _setup_engine(tmp_path)

    episodic.get_unconsolidated = AsyncMock(
        return_value=[
            _make_episode(id=1, importance=0.9, enrichment_status="pending"),
        ]
    )

    with _patch_embed(), _patch_cosine(0.3):
        result = await engine.run_consolidation(admin_user_id=12345)

    assert result.episodes_consolidated == 0
    episodic.mark_consolidated.assert_not_called()


# --- Phase 3: Consolidate ---


async def test_consolidate_creates_file_for_high_importance(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    episodes = [_make_episode(id=1, importance=0.9, content="важный инсайт")]
    state = await engine._phase_orient()

    with _patch_embed(), _patch_cosine(0.3):  # Low similarity → no match
        actions, _ = await engine._phase_consolidate(episodes, state, 12345)

    assert len(actions) == 1
    assert actions[0].action == "create"


async def test_consolidate_updates_existing_on_similarity(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    episodes = [_make_episode(id=1, content="I am Zhvusha and I learn")]
    state = await engine._phase_orient()

    with _patch_embed(), _patch_cosine(0.85):  # High similarity → match
        actions, _ = await engine._phase_consolidate(episodes, state, 12345)

    # Should find core.md as similar and update it
    assert any(a.action == "update" for a in actions)


async def test_consolidate_filters_sensitive_data(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(id=1, content="взял кредит 500000 рублей, никому не говори"),
    ]
    state = await engine._phase_orient()

    with _patch_embed(), _patch_cosine(0.3):
        actions, _ = await engine._phase_consolidate(episodes, state, 12345)

    # Sensitive content should be filtered out
    assert len(actions) == 0


# --- Phase 4: Prune ---


async def test_prune_keeps_under_200_lines(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    # Write a MEMORY.md with many lines
    lines = ["# Memory Index\n"] + [f"- entry {i}\n" for i in range(250)]
    engine.memory_index.write_text("".join(lines))

    # _phase_prune_index now writes to .pending/ (transactional)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    actions: list = []
    await engine._phase_prune_index(actions)

    pending_path = engine.pending_dir / "MEMORY.md"
    content = pending_path.read_text(encoding="utf-8")
    assert len(content.split("\n")) <= 200


# --- Transactional consolidation ---


async def test_pending_cleanup_on_next_run(tmp_path):
    engine, episodic, _ = _setup_engine(tmp_path)

    # Simulate incomplete previous run
    engine.pending_dir.mkdir(parents=True)
    (engine.pending_dir / "stale.md").write_text("incomplete")

    episodic.get_unconsolidated = AsyncMock(return_value=[])

    result = await engine.run_consolidation(admin_user_id=12345)
    # Should clean up .pending/ from previous run
    assert not engine.pending_dir.exists()
    # Phase 4 removed the early-return path — summary now reflects a full
    # (empty) run; the important invariant is that no episodes were processed.
    assert result.episodes_consolidated == 0


# --- Full run ---


async def test_full_consolidation_run(tmp_path):
    engine, episodic, _ws = _setup_engine(tmp_path)

    episodes = [
        _make_episode(id=1, importance=0.9, content="новый навык"),
    ]
    episodic.get_unconsolidated = AsyncMock(return_value=episodes)

    with _patch_embed(), _patch_cosine(0.3):
        result = await engine.run_consolidation(admin_user_id=12345)

    assert result.episodes_consolidated == 1
    assert len(result.files_created) > 0 or len(result.files_updated) > 0
    episodic.mark_consolidated.assert_awaited_once()


# --- Explicit rejection ---


async def test_explicit_rejection_writes_correction(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    with _patch_embed(), _patch_cosine(0.8):
        action = await engine.handle_explicit_rejection(
            rejected_conclusion="I am always right",
            nikita_correction="You should ask when unsure",
        )

    assert action is not None
    assert action.action == "update"
    assert "CORRECTED" in action.content

    # Diary entry should exist
    diary_files = list((engine.workspace / "diary").glob("*.md"))
    assert len(diary_files) == 1


# --- Contradiction synthesis ---


async def test_contradiction_synthesis_on_valence_conflict(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    # Episode with negative valence about a topic covered by core.md
    episodes = [
        _make_episode(
            id=1,
            content="короткие ответы плохо работают",
            valence="negative",
            embedding=[0.5] * 384,
        ),
    ]
    state = await engine._phase_orient()

    # Mock: high similarity to core.md, but _check_valence_conflict returns True
    with (
        _patch_embed(),
        _patch_cosine(0.85),
        patch.object(engine, "_check_valence_conflict", return_value=True),
    ):
        actions, partial = await engine._phase_consolidate(episodes, state, 12345)

    # Should have detected contradiction and synthesized
    assert len(partial.contradictions_found) == 1
    assert any("contradiction" in a.reason for a in actions)


async def test_contradiction_synthesis_calls_llm(tmp_path: Path):
    """When LLM router is available, synthesis uses strategist tier."""
    from src.llm.protocols import LLMResponse, LLMUsage

    engine, _, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(
            id=1,
            content="короткие ответы плохо работают",
            valence="negative",
            embedding=[0.5] * 384,
        ),
    ]
    state = await engine._phase_orient()

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(
            text="Я поняла, что длина ответа зависит от контекста.",
            model="sonnet",
            usage=LLMUsage(),
        )
    )

    with (
        _patch_embed(),
        _patch_cosine(0.85),
        patch.object(engine, "_check_valence_conflict", return_value=True),
        patch("src.llm.router.get_router", return_value=mock_router),
    ):
        actions, partial = await engine._phase_consolidate(episodes, state, 12345)

    # LLM should have been called with an LLMRequest
    mock_router.generate.assert_awaited_once()
    call = mock_router.generate.call_args
    request = call.args[0]
    assert request.tier == "strategist"
    assert request.reasoning_effort == "xhigh"
    assert "Предыдущий вывод" in request.prompt

    # Synthesized text should be in the action
    assert len(partial.contradictions_found) == 1
    assert any("зависит от контекста" in a.content for a in actions)


async def test_contradiction_synthesis_fallback_on_llm_error(tmp_path: Path):
    """When LLM fails, synthesis falls back to template."""
    from src.llm.protocols import LLMError

    engine, _, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(
            id=1,
            content="короткие ответы плохо работают",
            valence="negative",
            embedding=[0.5] * 384,
        ),
    ]
    state = await engine._phase_orient()

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(side_effect=LLMError("timeout"))

    with (
        _patch_embed(),
        _patch_cosine(0.85),
        patch.object(engine, "_check_valence_conflict", return_value=True),
        patch("src.llm.router.get_router", return_value=mock_router),
    ):
        actions, partial = await engine._phase_consolidate(episodes, state, 12345)

    # Should still produce a result via fallback
    assert len(partial.contradictions_found) == 1
    assert any("Предыдущее понимание" in a.content for a in actions)


# --- Three-factor reinforcement ---
# NOTE: `_find_reinforcement_triplets` was rewritten in Phase 5 to use
# enriched `intent="feedback"` + `feedback_strength` from metadata_json
# instead of hardcoded evaluative words. The legacy test that lived here
# exercised the old contract (`pattern` dict key, substring matching) and
# was removed as part of the schema migration. Phase 5 tests cover the
# new contract in `tests/test_reinforcement_triplets.py`.


# --- Pattern separation ---


async def test_pattern_separation_prevents_blind_merge(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    # Write existing file with "positive" content
    core = tmp_path / "workspace" / "personality" / "core.md"
    core.write_text("# Core\nМои ответы помогли клиенту успешно.\n")

    # Episode with opposite valence
    episodes = [
        _make_episode(
            id=1,
            content="Мой ответ не сработал",
            valence="negative",
            embedding=[0.5] * 384,
        ),
    ]
    state = await engine._phase_orient()

    with (
        _patch_embed(),
        _patch_cosine(0.85),
        patch.object(engine, "_check_valence_conflict", return_value=True),
    ):
        _actions, partial = await engine._phase_consolidate(episodes, state, 12345)

    # Should synthesize, not blindly append
    assert len(partial.contradictions_found) > 0


# --- Sensitive data protection ---


async def test_sensitive_financial_data_blocked(tmp_path):
    engine, _, _ = _setup_engine(tmp_path)

    episodes = [
        _make_episode(id=1, content="зарплата 150000 рублей в месяц"),
    ]
    state = await engine._phase_orient()

    with _patch_embed(), _patch_cosine(0.3):
        actions, _ = await engine._phase_consolidate(episodes, state, 12345)

    assert len(actions) == 0


# --- Phantom profile protection ---


async def test_people_mentioned_never_creates_profile(tmp_path):
    """Enrichment people_mentioned must NOT trigger profile creation.

    Only real Telegram user_ids from episodes should create profiles.
    This test ensures _phase_consolidate only calls record_interaction
    with episode.user_id, never with names from metadata/enrichment.
    """
    engine, _, _ = _setup_engine(tmp_path)

    # Episode from a stranger whose enrichment mentions "Коля"
    episodes = [
        _make_episode(
            id=1,
            user_id=999,
            importance=0.9,
            content="видел крутой проект от Коли на GitHub",
            metadata_json='{"enrichment": {"people_mentioned": ["Коля"]}}',
        ),
    ]
    state = await engine._phase_orient()

    # Track what user_ids were passed to record_interaction
    recorded_ids: list[int] = []
    engine.people = SimpleNamespace(
        record_interaction=lambda uid: recorded_ids.append(uid),
    )

    with _patch_embed(), _patch_cosine(0.3):
        await engine._phase_consolidate(episodes, state, admin_user_id=12345)

    # Only real user_id 999 should be recorded, never "Коля"
    assert recorded_ids == [999]
    # No string names should ever appear
    assert all(isinstance(uid, int) for uid in recorded_ids)
