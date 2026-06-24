"""Smoke test for Phase 2 StagingWriter — exercises the full routing,
dedup, size-warn, and correction-with-original-claim paths on a tmp
workspace, then prints the contents of both staging files.

Run with:

    .venv/bin/python scripts/smoke_staging_writer.py

Zero external dependencies (no postgres, no Sonnet). Useful for
verifying writer behavior in isolation.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.learning_staging import StagingWriter
from src.memory.sonnet_enricher import LearningSignal


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="zhvusha_smoke_staging_"))
    print(f"📂 temp workspace: {tmp_root}")

    try:
        staging_dir = tmp_root / "personality" / ".staging"
        writer = StagingWriter(staging_dir)

        signals = [
            # 1) strong rule → immediate
            LearningSignal(
                type="rule",
                statement="не писать формально в personal mode",
                scope="tone",
                confidence=0.92,
                apply_immediately=True,
            ),
            # 2) fact → immediate
            LearningSignal(
                type="fact",
                statement="Никита 5 лет на Kwork, это единственный доход",
                scope="personal_facts",
                confidence=0.95,
                apply_immediately=True,
            ),
            # 3) correction with original_claim → immediate
            LearningSignal(
                type="correction",
                statement="Kwork — единственный источник дохода, другого нет",
                scope="personal_facts",
                confidence=0.9,
                apply_immediately=True,
                original_claim="предполагала, что у Никиты есть основная работа",
            ),
            # 4) weak preference → pending
            LearningSignal(
                type="preference",
                statement="кажется Никита предпочитает краткие ответы",
                scope="preferences",
                confidence=0.6,
                apply_immediately=False,
            ),
            # 5) duplicate of (1) — should be deduped
            LearningSignal(
                type="rule",
                statement="не писать формально в personal mode",
                scope="tone",
                confidence=0.92,
                apply_immediately=True,
            ),
        ]

        for idx, signal in enumerate(signals, start=1):
            target = writer.append(
                signal,
                episode_id=100 + idx,
                chat_id=12345 if idx != 4 else None,
            )
            if target is None:
                print(f"\n[{idx}] ⏭  dedup/skipped: {signal.statement[:50]}")
            else:
                print(
                    f"\n[{idx}] ✅ wrote to {target.name}: "
                    f"type={signal.type} scope={signal.scope} "
                    f"strong={signal.apply_immediately and signal.confidence > 0.8}"
                )

        print(f"\n{'=' * 75}")
        print("📄 FINAL STATE — learnings_immediate.md:")
        print("=" * 75)
        immediate = staging_dir / "learnings_immediate.md"
        if immediate.exists():
            print(immediate.read_text(encoding="utf-8"))
        else:
            print("(file does not exist)")

        print(f"\n{'=' * 75}")
        print("📄 FINAL STATE — learnings_pending.md:")
        print("=" * 75)
        pending = staging_dir / "learnings_pending.md"
        if pending.exists():
            print(pending.read_text(encoding="utf-8"))
        else:
            print("(file does not exist)")

        print(f"\n{'=' * 75}")
        print("✅ All writes completed without errors.")
        print(f"   2 files written to: {staging_dir}")
        print("   Expected: 3 entries in immediate (signals 1/2/3, signal 5 deduped),")
        print("             1 entry in pending (signal 4).")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n🧹 cleaned up {tmp_root}")


if __name__ == "__main__":
    main()
