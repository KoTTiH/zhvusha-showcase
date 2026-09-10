"""Tests for Phase 4 — `ConsolidationEngine._phase_review_staging()`.

Kept in a separate file from `test_consolidation.py` so the Phase 4 suite is
independently auditable. Covers:

* Stale filter (>14 days auto-discard, pre-strategist)
* Strategist call via mocked `get_router`
* Decision application: promote_new / merge_existing / discard / hold
* Collision handling: promote into an existing file downgrades to merge
* Sensitive-data block on strategist outputs
* Error paths: LLM error, JSON parse fail, decision count mismatch,
  invalid target_file path
* Post-commit staging cleanup (rewrite held blocks, preserve tail writes)
* Integration with `run_consolidation`: drains staging when episodes empty,
  includes counters in summary, leaves staging intact on pre-commit failure
* `_coalesce_actions` dedupe-by-path with last-writer-wins
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

from src.llm.protocols import LLMResponse, LLMUsage
from src.memory.consolidation import (
    ConsolidationAction,
    ConsolidationEngine,
    StagingReviewResult,
)


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="opus", usage=LLMUsage())


if TYPE_CHECKING:
    from pathlib import Path


# --- Helpers (copy of test_consolidation.py helpers to keep tests isolated) ---


def _setup_engine(tmp_path: Path) -> tuple[ConsolidationEngine, AsyncMock, Path]:
    ws = tmp_path / "workspace"
    personality = ws / "personality"
    personality.mkdir(parents=True)
    (personality / "core.md").write_text("# Core\nI am Zhvusha.\n")
    (personality / "genes.md").write_text("# Genes\n")
    (personality / "MEMORY.md").write_text("# Memory Index\n")

    episodic = AsyncMock()
    episodic.get_unconsolidated = AsyncMock(return_value=[])
    episodic.mark_consolidated = AsyncMock()

    people = SimpleNamespace(record_interaction=lambda _uid: None)

    engine = ConsolidationEngine(episodic, ws, people)  # type: ignore[arg-type]
    return engine, episodic, ws


def _patch_embed() -> Any:
    return patch(
        "src.memory.consolidation.EmbeddingService.embed",
        return_value=[0.5] * 384,
    )


def _patch_cosine(val: float = 0.5) -> Any:
    return patch(
        "src.memory.consolidation.EmbeddingService.cosine_similarity",
        return_value=val,
    )


def _write_entry(
    file: Path,
    *,
    type_: str = "rule",
    scope: str = "tone",
    statement: str = "test statement",
    confidence: float = 0.9,
    episode_id: int = 1,
    chat_id: int | None = None,
    original_claim: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Append an entry to a staging file using the canonical format."""
    if timestamp is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M")
    file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## [{type_}] {scope} — {timestamp}",
        f"**Statement:** {statement}",
        f"**Confidence:** {confidence}",
    ]
    if chat_id is not None:
        lines.append(f"**Chat:** {chat_id}")
    lines.append(f"**Trigger episode:** {episode_id}")
    if original_claim is not None:
        lines.append(f"**Original claim:** {original_claim}")
    lines.append("")
    with file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _decision(
    entry_index: int,
    decision: str,
    target_file: str | None = None,
    final_text: str = "",
    reason: str = "test",
) -> dict[str, object]:
    return {
        "entry_index": entry_index,
        "decision": decision,
        "target_file": target_file,
        "final_text": final_text,
        "reason": reason,
    }


def _patch_router(raw: str) -> Any:
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw))
    return patch("src.llm.router.get_router", return_value=mock_router)


def _patch_router_error(exc: BaseException) -> Any:
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(side_effect=exc)
    return patch("src.llm.router.get_router", return_value=mock_router)


# --- Skip-empty / stale filter ---


async def test_review_skipped_when_staging_dir_missing(tmp_path: Path) -> None:
    engine, _, _ = _setup_engine(tmp_path)
    # No .staging/ dir created.
    with _patch_router("[]") as mock_get_router:
        review = await engine._phase_review_staging()
    assert review.skipped_empty is True
    assert review.actions == []
    mock_get_router.assert_not_called()


async def test_review_skipped_when_both_files_empty(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "learnings_immediate.md").write_text("   \n\n  ", encoding="utf-8")
    (staging / "learnings_pending.md").write_text("", encoding="utf-8")

    with _patch_router("[]") as mock_get_router:
        review = await engine._phase_review_staging()

    assert review.skipped_empty is True
    mock_get_router.assert_not_called()


async def test_review_stale_filter_drops_entries_older_than_14_days(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    old_ts = (datetime.now(tz=UTC) - timedelta(days=15)).strftime("%Y-%m-%d %H:%M")
    fresh_ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M")

    _write_entry(
        staging / "learnings_pending.md",
        statement="stale entry",
        episode_id=1,
        timestamp=old_ts,
    )
    _write_entry(
        staging / "learnings_pending.md",
        statement="fresh entry",
        episode_id=2,
        timestamp=fresh_ts,
    )

    raw = json.dumps([_decision(0, "discard")])
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw))
    with patch("src.llm.router.get_router", return_value=mock_router):
        review = await engine._phase_review_staging()

    assert review.stale_discarded == 1
    # Strategist was called with 1 fresh entry.
    mock_router.generate.assert_awaited_once()
    request_arg = mock_router.generate.call_args.args[0]
    prompt_arg = request_arg.prompt
    assert request_arg.tier == "strategist"
    assert request_arg.reasoning_effort == "xhigh"
    assert "fresh entry" in prompt_arg
    assert "stale entry" not in prompt_arg


async def test_review_keeps_two_week_batch_under_context_window(
    tmp_path: Path,
) -> None:
    """Two-week staging can review up to 300 signals in one strategist call."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    now = datetime.now(tz=UTC)
    for idx in range(300):
        ts = (now - timedelta(days=idx % 14)).strftime("%Y-%m-%d %H:%M")
        _write_entry(
            staging / "learnings_pending.md",
            statement=f"two week signal {idx}",
            episode_id=idx,
            timestamp=ts,
        )

    prep = engine._prepare_staging_batch(staging)
    prompt = engine._build_staging_review_prompt(prep.fresh, [])

    assert len(prep.fresh) == 300
    assert prep.initial_held == []
    assert "Entry 299" in prompt
    assert len(prompt) < 400_000


async def test_review_holds_overflow_beyond_two_week_context_batch(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M")
    for idx in range(301):
        _write_entry(
            staging / "learnings_pending.md",
            statement=f"overflow signal {idx}",
            episode_id=idx,
            timestamp=now,
        )

    prep = engine._prepare_staging_batch(staging)

    assert len(prep.fresh) == 300
    assert len(prep.initial_held) == 1
    assert "overflow signal 300" in prep.initial_held[0]


# --- Decision application ---


async def test_review_promote_new_creates_pending_file(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        type_="rule",
        scope="tone",
        statement="новое правило формальности",
        episode_id=100,
    )

    raw = json.dumps(
        [
            _decision(
                0,
                "promote_new",
                target_file="values/tone_rule.md",
                final_text="Не писать формально. Использовать расслабленный тон.",
            )
        ]
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.promoted == 1
    assert len(review.actions) == 1
    assert review.actions[0].action == "create"
    assert review.actions[0].file_path == "values/tone_rule.md"
    pending_file = engine.pending_dir / "values/tone_rule.md"
    assert pending_file.exists()
    content = pending_file.read_text(encoding="utf-8")
    assert "Не писать формально" in content


async def test_review_merge_existing_appends_to_pending_when_phase3_wrote_it(
    tmp_path: Path,
) -> None:
    """If Phase 3 already wrote .pending/core.md, Phase 4 merge must read
    from pending (not from personality) so it stacks on top of Phase 3."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        type_="rule",
        scope="tone",
        statement="расширение core",
        episode_id=50,
    )

    # Phase 3 already wrote .pending/core.md with an episode marker
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    phase3_content = "# Core\nI am Zhvusha.\n\n<!-- Episode 999 -->\nphase 3 text\n"
    (engine.pending_dir / "core.md").write_text(phase3_content, encoding="utf-8")

    raw = json.dumps(
        [
            _decision(
                0,
                "merge_existing",
                target_file="core.md",
                final_text="phase 4 addition",
            )
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.merged == 1
    merged = (engine.pending_dir / "core.md").read_text(encoding="utf-8")
    assert "phase 3 text" in merged
    assert "phase 4 addition" in merged


async def test_review_merge_existing_reads_from_personality_when_no_pending(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="merge into core",
        episode_id=10,
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(
        [
            _decision(
                0,
                "merge_existing",
                target_file="core.md",
                final_text="добавление из staging",
            )
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.merged == 1
    merged = (engine.pending_dir / "core.md").read_text(encoding="utf-8")
    assert "I am Zhvusha" in merged  # original content preserved
    assert "добавление из staging" in merged


async def test_review_promote_collision_downgrades_to_merge(tmp_path: Path) -> None:
    """If strategist says promote_new but the target already exists anywhere
    (personality/ or .pending/), we downgrade to merge."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="collides with core",
        episode_id=5,
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(
        [
            _decision(
                0,
                "promote_new",
                target_file="core.md",  # core.md exists in personality/
                final_text="new content colliding",
            )
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    # Counted as merge, not promote
    assert review.merged == 1
    assert review.promoted == 0
    assert review.actions[0].action == "update"


async def test_review_sensitive_filter_blocks_merge(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="факт про работу",
        episode_id=1,
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(
        [
            _decision(
                0,
                "merge_existing",
                target_file="core.md",
                final_text="взял кредит 500000 рублей, никому не говори",
            )
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.discarded == 1
    assert review.merged == 0
    # No .pending/ write
    assert not (engine.pending_dir / "core.md").exists()


async def test_review_discard_entry_is_dropped(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md", statement="weak signal", episode_id=1
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps([_decision(0, "discard", reason="too weak")])
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.discarded == 1
    assert review.held_blocks == []
    assert review.actions == []


async def test_review_hold_entry_preserved_in_held_blocks(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="нужно больше данных",
        episode_id=1,
    )
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps([_decision(0, "hold", reason="need more evidence")])
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.held == 1
    assert len(review.held_blocks) == 1
    assert "нужно больше данных" in review.held_blocks[0]


# --- Strategist prompt construction ---


async def test_review_excludes_memory_md_from_prompt(tmp_path: Path) -> None:
    """MEMORY.md must not appear in <EXISTING_FILES> — it's the index file,
    managed by _phase_prune_index, and is never a valid write target.
    Without this exclusion the reviewer could pick MEMORY.md as a dumping
    ground and the entry would get silently discarded by _SAFE_TARGET_RE."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="some fact", episode_id=1)

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=_llm_resp(json.dumps([_decision(0, "discard")]))
    )
    with patch("src.llm.router.get_router", return_value=mock_router):
        await engine._phase_review_staging()

    mock_router.generate.assert_awaited_once()
    request = mock_router.generate.call_args.args[0]
    prompt = request.prompt
    # The EXISTING_FILES block should list core.md and genes.md but NOT MEMORY.md.
    assert "<EXISTING_FILES>" in prompt
    files_section = prompt.split("<EXISTING_FILES>")[1].split("</EXISTING_FILES>")[0]
    assert "core.md" in files_section
    assert "genes.md" in files_section
    assert "MEMORY.md" not in files_section


async def test_review_excludes_reinforcements_md_from_prompt(
    tmp_path: Path,
) -> None:
    """reinforcements.md is an auto-managed Phase 5 file — reviewer must not be
    allowed to target it as a write target via staging review. Regression
    for the `_AUTO_MANAGED_FILES` filter added in Phase 5."""
    engine, _, ws = _setup_engine(tmp_path)
    # Pre-seed reinforcements.md in personality/
    (ws / "personality" / "reinforcements.md").write_text(
        "# Reinforcement Patterns\n\nsome content\n", encoding="utf-8"
    )
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="some fact", episode_id=1)

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=_llm_resp(json.dumps([_decision(0, "discard")]))
    )
    with patch("src.llm.router.get_router", return_value=mock_router):
        await engine._phase_review_staging()

    mock_router.generate.assert_awaited_once()
    request = mock_router.generate.call_args.args[0]
    prompt = request.prompt
    assert "<EXISTING_FILES>" in prompt
    files_section = prompt.split("<EXISTING_FILES>")[1].split("</EXISTING_FILES>")[0]
    assert "core.md" in files_section
    # Both auto-managed files are filtered out
    assert "MEMORY.md" not in files_section
    assert "reinforcements.md" not in files_section


# --- Error paths ---


async def test_review_llm_error_returns_review_failed(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="test", episode_id=1)

    with _patch_router_error(RuntimeError("CLI timeout")):
        review = await engine._phase_review_staging()

    assert review.review_failed is True
    assert review.actions == []
    # Staging file still exists — untouched
    assert (staging / "learnings_pending.md").exists()


async def test_review_bad_json_response_returns_review_failed(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="test", episode_id=1)

    with _patch_router("this is not json at all, just prose"):
        review = await engine._phase_review_staging()

    assert review.review_failed is True


async def test_review_decision_count_mismatch_returns_review_failed(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="a", episode_id=1)
    _write_entry(staging / "learnings_pending.md", statement="b", episode_id=2)
    _write_entry(staging / "learnings_pending.md", statement="c", episode_id=3)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Only 2 decisions for 3 entries
    raw = json.dumps(
        [
            _decision(0, "discard"),
            _decision(1, "discard"),
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    assert review.review_failed is True


async def test_review_invalid_target_file_path_rejected(tmp_path: Path) -> None:
    """Path traversal, absolute paths, non-.md extensions must be rejected."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(staging / "learnings_pending.md", statement="a", episode_id=1)
    _write_entry(staging / "learnings_pending.md", statement="b", episode_id=2)
    _write_entry(staging / "learnings_pending.md", statement="c", episode_id=3)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(
        [
            _decision(
                0,
                "promote_new",
                target_file="../etc/passwd",
                final_text="hack",
            ),
            _decision(1, "promote_new", target_file="/abs/path.md", final_text="hack"),
            _decision(2, "promote_new", target_file="foo.txt", final_text="hack"),
        ]
    )
    with _patch_router(raw):
        review = await engine._phase_review_staging()

    # All three rejected as discarded (unsafe path)
    assert review.discarded == 3
    assert review.promoted == 0
    assert review.actions == []


# --- Cleanup ---


async def test_cleanup_rewrites_pending_with_held_blocks(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    staging.mkdir(parents=True)

    # Pre-existing content was reviewed and consumed. Snapshot captures its
    # size so the cleanup considers everything up to that point as consumed.
    pending = staging / "learnings_pending.md"
    pending.write_text("reviewed and consumed content\n", encoding="utf-8")

    review = StagingReviewResult(
        held_blocks=[
            "\n## [rule] tone — 2026-04-04 09:00\n**Statement:** held A\n"
            "**Confidence:** 0.9\n**Trigger episode:** 1\n"
        ],
        held=1,
        snapshot_sizes={
            "learnings_immediate.md": 0,
            "learnings_pending.md": pending.stat().st_size,
        },
    )

    engine._apply_staging_cleanup(review)

    rewritten = pending.read_text(encoding="utf-8")
    assert "held A" in rewritten
    assert "reviewed and consumed" not in rewritten


async def test_cleanup_unlinks_immediate_when_no_tail_writes(tmp_path: Path) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    staging.mkdir(parents=True)
    immediate = staging / "learnings_immediate.md"
    immediate.write_text("consumed content\n", encoding="utf-8")

    review = StagingReviewResult(
        snapshot_sizes={
            "learnings_immediate.md": immediate.stat().st_size,
            "learnings_pending.md": 0,
        },
    )

    engine._apply_staging_cleanup(review)

    assert not immediate.exists()


async def test_cleanup_preserves_tail_writes_in_pending(tmp_path: Path) -> None:
    """A new entry appended between snapshot and cleanup must survive."""
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    staging.mkdir(parents=True)
    pending = staging / "learnings_pending.md"

    # Initial content that was parsed/reviewed
    initial = (
        "\n## [rule] tone — 2026-04-04 09:00\n"
        "**Statement:** reviewed\n"
        "**Confidence:** 0.9\n"
        "**Trigger episode:** 1\n"
    )
    pending.write_text(initial, encoding="utf-8")
    snapshot_size = pending.stat().st_size

    # A concurrent write lands after snapshot
    tail = (
        "\n## [fact] personal_facts — 2026-04-05 12:00\n"
        "**Statement:** tail write\n"
        "**Confidence:** 0.85\n"
        "**Trigger episode:** 99\n"
    )
    with pending.open("a", encoding="utf-8") as f:
        f.write(tail)

    review = StagingReviewResult(
        held_blocks=[],  # none held — reviewed entry was promoted/discarded
        snapshot_sizes={
            "learnings_immediate.md": 0,
            "learnings_pending.md": snapshot_size,
        },
    )
    engine._apply_staging_cleanup(review)

    # Tail write preserved
    assert pending.exists()
    rewritten = pending.read_text(encoding="utf-8")
    assert "tail write" in rewritten
    assert "reviewed" not in rewritten  # original consumed


# --- run_consolidation integration ---


async def test_run_consolidation_drains_staging_with_no_episodes(
    tmp_path: Path,
) -> None:
    """No new episodes but staging has entries → strategist still called, staging
    drained, mark_consolidated NOT called."""
    engine, episodic, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="сигнал без episode",
        episode_id=10,
    )

    raw = json.dumps([_decision(0, "discard")])
    with _patch_embed(), _patch_cosine(0.3), _patch_router(raw):
        result = await engine.run_consolidation(admin_user_id=12345)

    assert result.episodes_consolidated == 0
    assert result.staging_discarded == 1
    episodic.mark_consolidated.assert_not_called()
    # Staging drained
    assert not (staging / "learnings_pending.md").exists() or (
        (staging / "learnings_pending.md").read_text(encoding="utf-8").strip() == ""
    )


async def test_run_consolidation_skips_mark_consolidated_when_no_episodes(
    tmp_path: Path,
) -> None:
    """Regression: removing the early-return must not break no-episodes path."""
    engine, episodic, _ = _setup_engine(tmp_path)
    # No staging entries either.
    result = await engine.run_consolidation(admin_user_id=12345)
    assert result.episodes_consolidated == 0
    episodic.mark_consolidated.assert_not_called()


async def test_run_consolidation_summary_includes_staging_counters(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="entry for promote",
        episode_id=1,
    )
    _write_entry(
        staging / "learnings_pending.md",
        statement="entry for discard",
        episode_id=2,
    )

    raw = json.dumps(
        [
            _decision(
                0,
                "promote_new",
                target_file="values/promoted.md",
                final_text="promoted content",
            ),
            _decision(1, "discard", reason="weak"),
        ]
    )
    with _patch_embed(), _patch_cosine(0.3), _patch_router(raw):
        result = await engine.run_consolidation(admin_user_id=12345)

    assert result.staging_promoted == 1
    assert result.staging_discarded == 1
    assert "promoted" in result.summary.lower()


async def test_run_consolidation_failure_before_commit_leaves_staging_intact(
    tmp_path: Path,
) -> None:
    engine, _, ws = _setup_engine(tmp_path)
    staging = ws / "personality" / ".staging"
    _write_entry(
        staging / "learnings_pending.md",
        statement="untouched on failure",
        episode_id=1,
    )

    raw = json.dumps(
        [
            _decision(
                0,
                "promote_new",
                target_file="values/new.md",
                final_text="content",
            )
        ]
    )
    # Make _phase_prune_index raise so the pre-commit path fails
    with (
        _patch_embed(),
        _patch_cosine(0.3),
        _patch_router(raw),
        patch.object(
            engine,
            "_phase_prune_index",
            side_effect=RuntimeError("prune blew up"),
        ),
        contextlib.suppress(RuntimeError),
    ):
        await engine.run_consolidation(admin_user_id=12345)

    # Staging file untouched
    assert (staging / "learnings_pending.md").exists()
    content = (staging / "learnings_pending.md").read_text(encoding="utf-8")
    assert "untouched on failure" in content
    # .pending/ cleaned up
    assert not engine.pending_dir.exists()


# --- _coalesce_actions ---


def test_coalesce_actions_dedupes_by_path_keeps_last(tmp_path: Path) -> None:
    engine, _, _ = _setup_engine(tmp_path)

    first = ConsolidationAction(
        action="update", file_path="core.md", content="a", reason="first"
    )
    second = ConsolidationAction(
        action="update", file_path="core.md", content="b", reason="second"
    )
    unrelated = ConsolidationAction(
        action="create", file_path="other.md", content="o", reason="o"
    )
    coalesced = engine._coalesce_actions([first, second, unrelated])
    assert len(coalesced) == 2
    # Order is insertion order; last wins per-path
    by_path = {a.file_path: a for a in coalesced}
    assert by_path["core.md"].reason == "second"
    assert by_path["other.md"].reason == "o"
