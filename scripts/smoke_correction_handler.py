"""Smoke test for Phase 3 correction auto-handling — exercises the real
`ConsolidationEngine.handle_explicit_rejection()` method against a seeded
tmp workspace with REAL embeddings (sentence-transformers, not mocked).

Covers the gap in unit tests (`test_chat_response_enrichment.py` mocks
the engine entirely — we need to see if cosine similarity actually finds
the right file on realistic Russian text).

Run with:

    .venv/bin/python scripts/smoke_correction_handler.py

First call loads the sentence-transformers model (~2-5 sec, ~471MB).
Subsequent scenarios reuse the cached model (~5ms per embed).

Zero dependencies on Sonnet, Postgres, or Telegram. Requires only the
embedding model (already used by the main bot).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

from src.memory.consolidation import ConsolidationAction, ConsolidationEngine

_SEEDED_PROFILE = """# Никита

Никита работает full-time в компании "Рога и Копыта" уже 3 года.
Фриланс для него это хобби по выходным, основной доход — офисная зарплата.
"""

_SEEDED_CORE = """# Core
I am Zhvusha, a personal AI agent.
"""


def _build_engine(ws: Path) -> ConsolidationEngine:
    """Build a ConsolidationEngine against the seeded workspace. `handle_explicit_rejection`
    doesn't touch `self.episodic` or `self.people`, so stubs are sufficient."""
    episodic_stub = AsyncMock()
    people_stub = SimpleNamespace(record_interaction=lambda _uid: None)
    return ConsolidationEngine(episodic_stub, ws, people_stub)  # type: ignore[arg-type]


def _seed_workspace(ws: Path) -> None:
    (ws / "personality" / "nikita").mkdir(parents=True, exist_ok=True)
    (ws / "personality" / "nikita" / "profile.md").write_text(
        _SEEDED_PROFILE, encoding="utf-8"
    )
    (ws / "personality" / "core.md").write_text(_SEEDED_CORE, encoding="utf-8")


def _print_file(path: Path, label: str) -> None:
    print(f"\n📄 {label}: {path.relative_to(path.parents[1])}")
    print("-" * 75)
    if path.exists():
        print(path.read_text(encoding="utf-8"))
    else:
        print("(file does not exist)")
    print("-" * 75)


async def _scenario_match(engine: ConsolidationEngine, ws: Path) -> None:
    print("\n" + "=" * 75)
    print("SCENARIO 1 — expect match: rejected claim semantically close to profile.md")
    print("=" * 75)

    action: ConsolidationAction | None = await engine.handle_explicit_rejection(
        rejected_conclusion=(
            'Никита работает full-time в компании "Рога и Копыта", фриланс это хобби'
        ),
        nikita_correction=(
            "Никита НЕ работает в офисе, Kwork — единственный источник дохода "
            "уже 5 лет, никакой основной работы нет"
        ),
    )

    if action is None:
        print("❌ No match found (similarity < 0.5). Unexpected for this scenario.")
        return

    print(f"✅ Matched file:  {action.file_path}")
    print(f"   Action:        {action.action}")
    print(f"   Reason:        {action.reason}")
    _print_file(ws / "personality" / action.file_path, "MODIFIED FILE")

    diary_files = sorted((ws / "diary").glob("*.md"))
    if diary_files:
        _print_file(diary_files[0], "DIARY ENTRY")
    else:
        print("❌ No diary file created. Unexpected.")


async def _scenario_no_match(engine: ConsolidationEngine) -> None:
    print("\n" + "=" * 75)
    print(
        "SCENARIO 2 — expect NO match: rejected claim about an unrelated topic "
        "(similarity < 0.5)"
    )
    print("=" * 75)

    action = await engine.handle_explicit_rejection(
        rejected_conclusion="квантовая криптография на решётках Лидла",
        nikita_correction="на самом деле это про эллиптические кривые",
    )

    if action is None:
        print("✅ Correctly returned None — no personality file matches this topic.")
    else:
        print(
            f"❌ Unexpected match: {action.file_path}. "
            "Embeddings may be too permissive."
        )


async def _scenario_direct_strike(ws: Path) -> None:
    """Sanity: after SCENARIO 1 modified profile.md, the file should now contain
    the strikethrough + correction block. Re-read and verify markers."""
    print("\n" + "=" * 75)
    print("SCENARIO 3 — post-mutation sanity: verify markers written")
    print("=" * 75)

    profile = (ws / "personality" / "nikita" / "profile.md").read_text(encoding="utf-8")
    checks = [
        ("<!-- CORRECTED -->", "CORRECTED comment marker"),
        ("~~", "strikethrough marker"),
        ("**Correction:**", "correction label"),
        ("Kwork", "correction body"),
    ]
    all_ok = True
    for marker, label in checks:
        ok = marker in profile
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {label:<30} ({marker!r})")
        all_ok = all_ok and ok

    if all_ok:
        print("\n✅ All markers present — correction applied correctly.")
    else:
        print("\n❌ Some markers missing — correction format may have drifted.")


async def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="zhvusha_smoke_correction_"))
    print(f"📂 temp workspace: {tmp_root}")
    print("⏳ loading sentence-transformers model on first embed (~2-5 sec)...")

    try:
        _seed_workspace(tmp_root)
        engine = _build_engine(tmp_root)

        await _scenario_match(engine, tmp_root)
        await _scenario_no_match(engine)
        await _scenario_direct_strike(tmp_root)

        print("\n" + "=" * 75)
        print("✅ All scenarios executed. Review output above for correctness.")
        print("=" * 75)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n🧹 cleaned up {tmp_root}")


if __name__ == "__main__":
    asyncio.run(main())
