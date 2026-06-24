"""One-time migration of file-based knowledge to PostgreSQL.

Usage:
    python scripts/migrate_knowledge_files.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def main() -> None:
    from src.core.config import get_settings
    from src.knowledge.store import KnowledgeStore
    from src.memory.database import get_engine, get_session_maker

    settings = get_settings()
    knowledge_dir = Path(settings.workspace_path).expanduser() / "knowledge"

    if not knowledge_dir.exists():
        print(f"Knowledge directory not found: {knowledge_dir}")
        return

    engine = get_engine(settings.database_url)
    sm = get_session_maker(engine)
    store = KnowledgeStore(sm)

    migrated = 0
    skipped = 0

    for md_file in sorted(knowledge_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Title from first line
        title = lines[0].lstrip("# ").strip() if lines else md_file.stem
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        if not body:
            skipped += 1
            continue

        # Category from directory name
        category = md_file.parent.name
        category_path = None if category == "knowledge" else category

        try:
            entry_id = await store.add_entry(
                title=title,
                content=body,
                category_path=category_path,
                source="file_migration",
            )
            print(f"  Migrated: {md_file.name} → #{entry_id}")
            migrated += 1
        except Exception as e:
            print(f"  ERROR: {md_file.name} → {e}")
            skipped += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
