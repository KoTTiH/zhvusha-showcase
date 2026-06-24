"""End-to-end smoke test: record episode → enrich via Sonnet → read back.

Exercises the full production path against a live postgres. Run:

    .venv/bin/python scripts/smoke_enrichment_db.py

Requires: docker compose up -d, alembic upgrade head, .env configured.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.core.config import get_settings
from src.memory.database import Episode, get_engine, get_session_maker
from src.memory.episodic import EpisodicMemory
from src.memory.sonnet_enricher import SonnetEnricher


async def main() -> None:
    settings = get_settings()
    engine = get_engine(settings.database_url)
    session_maker = get_session_maker(engine)
    memory = EpisodicMemory(session_maker, admin_user_id=settings.admin_user_id)
    enricher = SonnetEnricher()

    test_message = "да заебала формальными советами, я 5 лет на kwork"
    print(f"\n📝 Recording episode: {test_message!r}")

    episode_id = await memory.record(
        content=test_message,
        user_id=settings.admin_user_id,
        chat_type="personal",
        role="user",
        source="chat",
    )
    print(f"   → episode_id={episode_id}")

    # Read placeholder state
    async with session_maker() as session:
        result = await session.execute(select(Episode).where(Episode.id == episode_id))
        ep = result.scalars().one()
        print("\n🟡 Placeholder state (before enrichment):")
        print(f"   importance={ep.importance}")
        print(f"   valence={ep.valence}")
        print(f"   confidence={ep.confidence}")
        print(f"   intent={ep.intent}")
        print(f"   emotion={ep.emotion}")

    print("\n🤖 Calling Sonnet enricher...")
    enrichment = await enricher.enrich(message=test_message)
    if enrichment is None:
        print("   ❌ enricher returned None")
        await engine.dispose()
        return

    print("\n✨ Enrichment result:")
    print(f"   importance={enrichment.importance}")
    print(f"   valence={enrichment.valence}")
    print(f"   intent={enrichment.intent}")
    print(f"   emotion={enrichment.emotion}")
    print(f"   confidence={enrichment.confidence}")
    print(f"   reasoning={enrichment.reasoning}")

    await memory.update_enrichment(episode_id, enrichment)
    print("\n✅ update_enrichment applied")

    # Re-read from DB to verify persistence
    async with session_maker() as session:
        result = await session.execute(select(Episode).where(Episode.id == episode_id))
        ep = result.scalars().one()
        print("\n🟢 Final DB state (after enrichment):")
        print(f"   importance={ep.importance}")
        print(f"   valence={ep.valence}")
        print(f"   confidence={ep.confidence}")
        print(f"   intent={ep.intent}")
        print(f"   emotion={ep.emotion}")

        # Verification assertions
        assert ep.importance == enrichment.importance
        assert ep.valence == enrichment.valence
        assert ep.intent == enrichment.intent
        assert ep.emotion == enrichment.emotion
        assert ep.confidence == enrichment.confidence
        assert ep.content == test_message  # content untouched
        print("\n✅ All fields match. Content preserved.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
