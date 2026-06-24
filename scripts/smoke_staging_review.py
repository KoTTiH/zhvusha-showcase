"""Smoke test for Phase 4 — Morning Staging Review.

Exercises the real `ConsolidationEngine._phase_review_staging()` end-to-end
against a seeded tmp workspace with the configured strategist tier. Covers
the gap in unit tests (`test_consolidation_staging_review.py` mocks the router
— we need to see whether the live strategist model actually returns valid JSON
on realistic Russian staging entries).

Run with:

    .venv/bin/python scripts/smoke_staging_review.py

Requires:
- Codex CLI in PATH by default (`CODEX_CLI_PATH`, subscription auth).
- ~30-60 seconds per run (one strategist call via subprocess/API adapter).
- Negligible prompt size (~3-5KB prompt, ~1KB response).

Zero dependencies on Postgres or Telegram. Uses stub episodic/people.
"""

# ruff: noqa: I001

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.consolidation import ConsolidationEngine, StagingReviewResult


_CORE_MD = """# Core

Я — Жвуша, персональный AI-агент Никиты. Помогаю с Kwork, развитием, кодом.
"""

_GENES_MD = """# Genes

| Gene | Value |
|---|---|
| Curiosity | 0.8 |
| Directness | 0.9 |
"""

_MEMORY_INDEX = """# Memory Index

- [core.md](core.md) — кто я
- [genes.md](genes.md) — базовые характеристики
"""


def _build_engine(ws: Path) -> ConsolidationEngine:
    """Build a ConsolidationEngine against the seeded workspace.
    `_phase_review_staging` doesn't touch `self.episodic` or `self.people`,
    so stubs are sufficient."""
    episodic_stub = AsyncMock()
    episodic_stub.get_unconsolidated = AsyncMock(return_value=[])
    episodic_stub.mark_consolidated = AsyncMock()
    people_stub = SimpleNamespace(record_interaction=lambda _uid: None)
    return ConsolidationEngine(episodic_stub, ws, people_stub)  # type: ignore[arg-type]


def _seed_personality(ws: Path) -> None:
    personality = ws / "personality"
    personality.mkdir(parents=True, exist_ok=True)
    (personality / "core.md").write_text(_CORE_MD, encoding="utf-8")
    (personality / "genes.md").write_text(_GENES_MD, encoding="utf-8")
    (personality / "MEMORY.md").write_text(_MEMORY_INDEX, encoding="utf-8")


def _write_entry(
    file: Path,
    *,
    type_: str,
    scope: str,
    statement: str,
    confidence: float,
    episode_id: int,
    timestamp: str,
    chat_id: int | None = 12345,
    original_claim: str | None = None,
) -> None:
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


def _seed_staging(ws: Path) -> None:
    """Seed 4 entries: strong rule, weak preference, fact, and one stale."""
    staging = ws / "personality" / ".staging"
    now = datetime.now(tz=UTC)
    fresh_ts = now.strftime("%Y-%m-%d %H:%M")
    stale_ts = (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M")

    pending = staging / "learnings_pending.md"
    immediate = staging / "learnings_immediate.md"

    # 1) Strong rule — strategist should promote_new
    _write_entry(
        pending,
        type_="rule",
        scope="tone",
        statement="не писать формально в personal mode, использовать расслабленный тон на 'ты'",
        confidence=0.92,
        episode_id=1001,
        timestamp=fresh_ts,
    )
    # 2) Weak one-off — strategist should discard
    _write_entry(
        pending,
        type_="preference",
        scope="preferences",
        statement="сегодня у Никиты было плохое настроение, просил короткие ответы",
        confidence=0.45,
        episode_id=1002,
        timestamp=fresh_ts,
    )
    # 3) Solid fact — strategist should promote_new or merge into personal_facts
    _write_entry(
        pending,
        type_="fact",
        scope="personal_facts",
        statement="Никита 5 лет на Kwork, это единственный источник дохода",
        confidence=0.95,
        episode_id=1003,
        timestamp=fresh_ts,
    )
    # 4) STALE — should be auto-discarded before the strategist is even called
    _write_entry(
        immediate,
        type_="rule",
        scope="tone",
        statement="stale правило которое должно быть удалено по возрасту",
        confidence=0.9,
        episode_id=999,
        timestamp=stale_ts,
    )


def _print_file(path: Path, label: str) -> None:
    print(f"\n📄 {label}: {path.name}")
    print("-" * 75)
    if path.exists():
        print(path.read_text(encoding="utf-8"))
    else:
        print("(file does not exist)")
    print("-" * 75)


def _print_review(review: StagingReviewResult) -> None:
    print("\n" + "=" * 75)
    print("STAGING REVIEW RESULT")
    print("=" * 75)
    print(f"  skipped_empty:     {review.skipped_empty}")
    print(f"  review_failed:     {review.review_failed}")
    print(f"  promoted:          {review.promoted}")
    print(f"  merged:            {review.merged}")
    print(f"  discarded:         {review.discarded}")
    print(f"  held:              {review.held}")
    print(f"  stale_discarded:   {review.stale_discarded}")
    print(f"  parse_failed:      {review.parse_failed}")
    print(f"  actions count:     {len(review.actions)}")
    print(f"  held blocks count: {len(review.held_blocks)}")
    print(f"  snapshot sizes:    {review.snapshot_sizes}")

    if review.actions:
        print("\n  ACTIONS:")
        for a in review.actions:
            print(f"    - {a.action:6s} {a.file_path:40s} reason: {a.reason[:50]}")

    if review.held_blocks:
        print("\n  HELD BLOCKS:")
        for i, block in enumerate(review.held_blocks, 1):
            print(f"    [{i}] {block[:200]}")


async def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="zhvusha_smoke_phase4_"))
    print(f"📂 temp workspace: {tmp_root}")
    print("⏳ calling configured strategist tier (~30-60 sec)...\n")

    try:
        _seed_personality(tmp_root)
        _seed_staging(tmp_root)

        print("SEEDED STAGING:")
        _print_file(
            tmp_root / "personality" / ".staging" / "learnings_pending.md",
            "pending (fresh entries)",
        )
        _print_file(
            tmp_root / "personality" / ".staging" / "learnings_immediate.md",
            "immediate (1 stale)",
        )

        engine = _build_engine(tmp_root)
        # _phase_review_staging writes to .pending/, so it must exist.
        engine.pending_dir.mkdir(parents=True, exist_ok=True)

        review = await engine._phase_review_staging()
        _print_review(review)

        if review.review_failed:
            print("\n❌ REVIEW FAILED — strategist call or JSON parse errored.")
            print(
                "   Check logs for staging_review_llm_failed / staging_review_parse_failed."
            )
            return

        # Walk .pending/ to show what would be committed
        pending_files = sorted(engine.pending_dir.rglob("*.md"))
        if pending_files:
            print("\n" + "=" * 75)
            print(f"PENDING FILES ({len(pending_files)}) — ready to commit:")
            print("=" * 75)
            for pf in pending_files:
                rel = pf.relative_to(engine.pending_dir)
                _print_file(pf, f".pending/{rel}")
        else:
            print("\n(no .pending/ files written)")

        # Simulate post-commit cleanup
        print("\n" + "=" * 75)
        print("SIMULATING _apply_staging_cleanup (post-commit)")
        print("=" * 75)
        engine._apply_staging_cleanup(review)

        staging_dir = tmp_root / "personality" / ".staging"
        _print_file(
            staging_dir / "learnings_pending.md",
            "AFTER cleanup — learnings_pending.md",
        )
        _print_file(
            staging_dir / "learnings_immediate.md",
            "AFTER cleanup — learnings_immediate.md",
        )

        print("\n" + "=" * 75)
        print("✅ Phase 4 smoke complete. Review output above for correctness.")
        print("=" * 75)
        print("\nExpected outcomes:")
        print("  - stale_discarded == 1 (the 10-day-old entry)")
        print("  - strategist sees 3 fresh entries and returns 3 decisions")
        print("  - Strong rule → promote_new or merge into existing tone file")
        print("  - Weak one-off → discard")
        print("  - Solid fact → promote_new or merge into personal_facts")
        print("  - After cleanup: staging files drained (unless entries were held)")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n🧹 cleaned up {tmp_root}")


if __name__ == "__main__":
    asyncio.run(main())
