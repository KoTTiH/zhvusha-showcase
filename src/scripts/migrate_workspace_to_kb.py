"""Migrate external knowledge from workspace files to Knowledge Base.

Moves files matching the mapping below into KB entries, then archives
the originals under workspace/archive/migrated/.

Usage:
    python -m src.scripts.migrate_workspace_to_kb          # dry-run (default)
    python -m src.scripts.migrate_workspace_to_kb --apply  # actually migrate
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MigrationRule:
    """Maps a workspace glob pattern to a KB category."""

    glob_pattern: str
    category_path: str
    extra_tags: list[str] = field(default_factory=list)
    title_fn: str = "filename"  # "filename" or "parent_dir"


# Mapping: workspace path pattern → KB category
RULES: list[MigrationRule] = [
    MigrationRule(
        glob_pattern="knowledge/channels/*/summary.md",
        category_path="intel.channels",
        title_fn="parent_dir",
    ),
    MigrationRule(
        glob_pattern="knowledge/browser/*.md",
        category_path="intel.browser",
    ),
    MigrationRule(
        glob_pattern="knowledge/youtube/*.md",
        category_path="intel.youtube",
    ),
    MigrationRule(
        glob_pattern="knowledge/research/*.md",
        category_path="research",
    ),
]


def _get_title(file: Path, rule: MigrationRule) -> str:
    """Derive entry title from file path."""
    if rule.title_fn == "parent_dir":
        return f"Channel {file.parent.name} summary"
    return file.stem.replace("_", " ").replace("-", " ").strip().title()


async def run_migration(*, apply: bool = False) -> None:  # noqa: C901
    """Scan workspace and migrate matching files to KB."""
    import os

    from src.knowledge.store import KnowledgeStore
    from src.memory.database import get_engine, get_session_maker

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    workspace_path = os.environ.get("WORKSPACE_PATH", "~/zhvusha-workspace")
    ws = Path(workspace_path).expanduser()
    if not ws.exists():
        print(f"ERROR: Workspace not found: {ws}")
        return

    archive_root = ws / "archive" / "migrated"
    engine = get_engine(database_url)
    session_maker = get_session_maker(engine)
    store = KnowledgeStore(session_maker)

    total = 0
    migrated = 0

    for rule in RULES:
        files = sorted(ws.glob(rule.glob_pattern))
        if not files:
            continue

        print(f"\n--- {rule.category_path} ({rule.glob_pattern}) ---")
        for file in files:
            total += 1
            content = file.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                print(f"  SKIP (empty): {file.relative_to(ws)}")
                continue

            title = _get_title(file, rule)
            rel = file.relative_to(ws)
            tags = list(rule.extra_tags)

            print(
                f"  {'MIGRATE' if apply else 'DRY-RUN'}: {rel} → {rule.category_path} [{title}]"
            )

            if apply:
                try:
                    entry_id = await store.add_entry(
                        title=title,
                        content=content,
                        category_path=rule.category_path,
                        tags=tags,
                        source="migration",
                        content_type="fact",
                    )
                except Exception as e:
                    print(f"    ERROR: DB insert failed: {e}")
                    continue
                print(f"    → KB entry #{entry_id}")

                # Archive the original file
                archive_path = archive_root / rel
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(file), str(archive_path))
                except OSError as e:
                    print(
                        f"    ⚠ archive failed: {e} (KB entry created, original kept)"
                    )
                    continue
                print(f"    → archived to {archive_path.relative_to(ws)}")

            migrated += 1

    print(f"\n{'=' * 40}")
    print(f"Total files scanned: {total}")
    print(f"Files {'migrated' if apply else 'to migrate'}: {migrated}")
    if not apply and migrated > 0:
        print("\nRun with --apply to execute migration.")

    await engine.dispose()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate workspace files to Knowledge Base"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually migrate (default: dry-run)"
    )
    args = parser.parse_args()
    asyncio.run(run_migration(apply=args.apply))


if __name__ == "__main__":
    main()
