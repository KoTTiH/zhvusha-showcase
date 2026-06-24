from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from src.skills.workspace_session.collector import collect_inbox

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Create workspace root with inbox/ and logs/ dirs."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    return tmp_path


@pytest.fixture
def inbox_dir(workspace_root: Path) -> Path:
    return workspace_root / "inbox"


@pytest.fixture(autouse=True)
def _mock_phase3():
    """Skip Phase 3 collectors in unit tests."""
    with patch(
        "src.skills.workspace_session.collector.collect_phase3_sources",
        return_value=[],
    ):
        yield


async def test_collect_inbox_creates_file(inbox_dir: Path):
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())

    today = date(2026, 3, 31)
    await collect_inbox(inbox_dir, redis=redis, today=today)

    expected = inbox_dir / "2026-03-31.md"
    assert expected.is_file()


async def test_collect_inbox_omits_kwork_section_and_seen_ids(inbox_dir: Path):
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value={b"101", b"202"})

    today = date(2026, 3, 31)
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "## Kwork Projects" not in text
    assert "Seen project IDs" not in text
    redis.smembers.assert_not_awaited()


async def test_collect_inbox_contains_chat_section(inbox_dir: Path):
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())

    today = date(2026, 3, 31)
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "Chat" in text or "chat" in text


async def test_collect_inbox_no_redis(inbox_dir: Path):
    """When Redis is unavailable, still creates the file."""
    today = date(2026, 3, 31)
    await collect_inbox(inbox_dir, redis=None, today=today)

    expected = inbox_dir / "2026-03-31.md"
    assert expected.is_file()
    text = expected.read_text()
    assert "## Kwork Projects" not in text


async def test_collect_inbox_does_not_overwrite(inbox_dir: Path):
    """If today's inbox already exists, skip collection."""
    today = date(2026, 3, 31)
    existing = inbox_dir / "2026-03-31.md"
    existing.write_text("already collected")

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value={b"999"})

    await collect_inbox(inbox_dir, redis=redis, today=today)

    assert existing.read_text() == "already collected"


async def test_collect_inbox_redis_error_handled(inbox_dir: Path):
    """Redis errors should not crash collection."""
    redis = AsyncMock()
    redis.smembers = AsyncMock(side_effect=ConnectionError("no redis"))

    today = date(2026, 3, 31)
    await collect_inbox(inbox_dir, redis=redis, today=today)

    expected = inbox_dir / "2026-03-31.md"
    assert expected.is_file()


async def test_collect_inbox_reads_chat_log(workspace_root: Path):
    """Chat log from yesterday should appear in inbox."""
    inbox_dir = workspace_root / "inbox"

    today = date(2026, 3, 31)
    yesterday = today - timedelta(days=1)
    chat_dir = workspace_root / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    log_file = chat_dir / f"chat_{yesterday.isoformat()}.jsonl"
    entry = {
        "ts": "2026-03-30T10:00:00+00:00",
        "role": "user",
        "user_id": 12345,
        "text": "привет жвуша",
        "mode": "personal",
    }
    log_file.write_text(json.dumps(entry, ensure_ascii=False) + "\n")

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "привет жвуша" in text


async def test_collect_inbox_reads_today_chat_log(workspace_root: Path):
    """Ad-hoc /morning runs include today's chat log too."""
    inbox_dir = workspace_root / "inbox"

    today = date(2026, 3, 31)
    chat_dir = workspace_root / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    log_file = chat_dir / f"chat_{today.isoformat()}.jsonl"
    entry = {
        "ts": "2026-03-31T10:00:00+00:00",
        "role": "user",
        "user_id": 12345,
        "text": "сегодняшний самокодинг",
        "mode": "personal",
    }
    log_file.write_text(json.dumps(entry, ensure_ascii=False) + "\n")

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "сегодняшний самокодинг" in text


async def test_collect_inbox_no_chat_log(workspace_root: Path):
    """Missing chat log should not crash and show 'No chat activity'."""
    inbox_dir = workspace_root / "inbox"

    today = date(2026, 3, 31)
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "No chat activity" in text


async def test_collect_inbox_reads_promotions(workspace_root: Path):
    """Promotions flag file should appear in inbox People section."""
    inbox_dir = workspace_root / "inbox"
    inbox_dir.mkdir(exist_ok=True)

    # Create promotions flag
    flag = inbox_dir / "promotions.md"
    flag.write_text("- Stranger 99999 promoted to known (3+ interactions)\n")

    # Create people dir
    (workspace_root / "memory" / "people").mkdir(parents=True, exist_ok=True)

    today = date(2026, 3, 31)
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "99999" in text
    assert "promoted" in text.lower()

    # Flag file should be cleaned up
    assert not flag.exists()


async def test_collect_inbox_new_contacts(workspace_root: Path):
    """New people from yesterday should appear in People section."""
    inbox_dir = workspace_root / "inbox"

    # Create a profile with first_seen = yesterday
    people_dir = workspace_root / "memory" / "people" / "55555"
    people_dir.mkdir(parents=True)
    (people_dir / "profile.md").write_text(
        "---\nuser_id: 55555\nusername: newguy\n"
        "first_seen: 2026-03-30T10:00:00+00:00\n---\n"
    )

    today = date(2026, 3, 31)
    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=today)

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "newguy" in text
    assert "New contacts" in text


async def test_collect_inbox_skips_self_coding_archive_by_default(
    workspace_root: Path,
) -> None:
    """Temporary consolidation path must not ingest self-coding history."""
    inbox_dir = workspace_root / "inbox"
    node_dir = workspace_root / "self_coding_archive" / "greeting-abc123"
    node_dir.mkdir(parents=True)
    (node_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "slug: greeting-abc123",
                "spec_slug: greeting-calibration",
                "status: committed",
                "created_at: '2026-03-31T09:00:00+00:00'",
                "commit_sha: abc1234567890",
            ]
        ),
        encoding="utf-8",
    )

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(inbox_dir, redis=redis, today=date(2026, 3, 31))

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "Self-Coding Archive" not in text
    assert "greeting-calibration" not in text


async def test_collect_inbox_can_include_self_coding_archive_when_enabled(
    workspace_root: Path,
) -> None:
    """Explicit opt-in still carries Жвуша's cycle archive and chat context."""
    inbox_dir = workspace_root / "inbox"
    node_dir = workspace_root / "self_coding_archive" / "greeting-abc123"
    node_dir.mkdir(parents=True)
    (node_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "slug: greeting-abc123",
                "spec_slug: greeting-calibration",
                "status: committed",
                "created_at: '2026-03-31T09:00:00+00:00'",
                "commit_sha: abc1234567890",
                "metadata:",
                "  self_coding_actor: zhvusha",
                "  agent_backend: codex_cli",
            ]
        ),
        encoding="utf-8",
    )
    (node_dir / "insight.md").write_text(
        "# greeting-abc123\n\n## Вывод\nПриветствия откалиброваны.\n",
        encoding="utf-8",
    )
    (node_dir / "chat_context.md").write_text(
        "# Контекст /самокодинг\n\n- Никита: Жвуша переигрывает живость\n",
        encoding="utf-8",
    )

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(
        inbox_dir,
        redis=redis,
        today=date(2026, 3, 31),
        include_self_coding_archive=True,
    )

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "Self-Coding Archive" in text
    assert "greeting-calibration" in text
    assert "actor=zhvusha" in text
    assert "Жвуша переигрывает живость" in text


async def test_collect_inbox_marks_legacy_self_coding_archive(
    workspace_root: Path,
) -> None:
    """Morning inbox keeps actor/backend labels for older cycle nodes."""
    inbox_dir = workspace_root / "inbox"
    node_dir = workspace_root / "self_coding_archive" / "legacy-greeting-abc123"
    node_dir.mkdir(parents=True)
    (node_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "slug: legacy-greeting-abc123",
                "spec_slug: greeting-calibration",
                "status: committed",
                "created_at: '2026-03-31T09:00:00+00:00'",
                "commit_sha: abc1234567890",
                "model_config:",
                "  backend: codex_cli",
                "  executor: codex_cli",
                "tags:",
                "- self-coding",
            ]
        ),
        encoding="utf-8",
    )

    redis = AsyncMock()
    redis.smembers = AsyncMock(return_value=set())
    await collect_inbox(
        inbox_dir,
        redis=redis,
        today=date(2026, 3, 31),
        include_self_coding_archive=True,
    )

    text = (inbox_dir / "2026-03-31.md").read_text()
    assert "Self-Coding Archive" in text
    assert "legacy-greeting-abc123" in text
    assert "actor=zhvusha" in text
    assert "backend=codex_cli" in text
