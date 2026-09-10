"""Tests for knowledge base manager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from src.knowledge.manager import KnowledgeManager


@pytest.fixture
def knowledge_root(tmp_path: Path) -> Path:
    """Create knowledge directory structure."""
    knowledge = tmp_path / "knowledge"
    for sub in ("youtube", "channels", "browser", "research", "projects", "web"):
        (knowledge / sub).mkdir(parents=True)
    return knowledge


async def test_save_new_knowledge_creates_file(knowledge_root: Path):
    """Saving a new topic creates a markdown file."""
    mgr = KnowledgeManager(knowledge_root.parent)
    path = await mgr.save_knowledge(
        topic="aiogram 3 new features",
        content="Middleware chaining, FSM improvements, RouteGroup",
        source="youtube",
        category="youtube",
    )

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "aiogram 3" in content
    assert "Middleware chaining" in content


async def test_save_similar_topic_updates_existing(
    knowledge_root: Path, mock_embedding_service: list[float]
):
    """Saving a similar topic (cosine > 0.75) updates existing file."""
    mgr = KnowledgeManager(knowledge_root.parent)

    # Create initial entry
    await mgr.save_knowledge(
        topic="VPN setup guide",
        content="VLESS Reality configuration steps",
        source="youtube",
        category="youtube",
    )

    # Save similar topic — should update, not create new
    with patch(
        "src.knowledge.manager.EmbeddingService.cosine_similarity",
        return_value=0.85,
    ):
        mock_llm = AsyncMock(return_value="Merged: VLESS Reality + new methods")
        with patch.object(mgr, "_merge_via_llm", mock_llm):
            path = await mgr.save_knowledge(
                topic="VPN VLESS guide updated",
                content="New CDN bypass method for VLESS",
                source="channel",
                category="youtube",
            )

    content = path.read_text(encoding="utf-8")
    assert "Merged" in content


async def test_search_returns_relevant_entries(knowledge_root: Path):
    """Search finds relevant knowledge entries."""
    mgr = KnowledgeManager(knowledge_root.parent)

    # Create some entries
    await mgr.save_knowledge(
        topic="Python async patterns",
        content="asyncio, aiohttp, structured concurrency",
        source="browser",
        category="research",
    )
    await mgr.save_knowledge(
        topic="Kwork pricing strategy",
        content="Min 5000 for bots, 7000 for sites",
        source="chat",
        category="projects",
    )

    results = await mgr.search("python async", limit=5)
    assert len(results) >= 1
    assert any("Python" in r.topic or "async" in r.content for r in results)


async def test_get_relevant_for_context_respects_limit(knowledge_root: Path):
    """Context loading respects the token limit."""
    mgr = KnowledgeManager(knowledge_root.parent)

    # Create entries with substantial content
    for i in range(5):
        await mgr.save_knowledge(
            topic=f"Topic {i}",
            content=f"Content for topic {i} " * 50,
            source="test",
            category="research",
        )

    context = await mgr.get_relevant_for_context(current_topics=["Topic"], limit=3)
    # Should return something but not exceed reasonable size
    assert len(context) > 0
    assert len(context) < 5000  # Roughly ~500 tokens


async def test_cleanup_removes_stale_entries(knowledge_root: Path):
    """Cleanup removes entries not accessed in 90 days."""
    mgr = KnowledgeManager(knowledge_root.parent)

    # Create old file
    old_file = knowledge_root / "research" / "old_topic.md"
    old_file.write_text("# Old Topic\nOutdated content", encoding="utf-8")
    # Set mtime to 100 days ago
    import os

    old_time = (datetime.now(tz=UTC) - timedelta(days=100)).timestamp()
    os.utime(old_file, (old_time, old_time))

    # Create recent file
    new_file = knowledge_root / "research" / "new_topic.md"
    new_file.write_text("# New Topic\nFresh content", encoding="utf-8")

    removed = await mgr.cleanup_stale(max_age_days=90)

    assert "old_topic.md" in str(removed)
    assert not old_file.exists()
    assert new_file.exists()


async def test_handles_empty_knowledge_directory(tmp_path: Path):
    """Works correctly with empty knowledge directory."""
    mgr = KnowledgeManager(tmp_path)
    results = await mgr.search("anything", limit=5)
    assert results == []


async def test_records_episode_on_save(knowledge_root: Path, mock_episodic: AsyncMock):
    """Saving knowledge records an episode for searchability."""
    mgr = KnowledgeManager(knowledge_root.parent, episodic=mock_episodic)

    await mgr.save_knowledge(
        topic="Test knowledge",
        content="Some content",
        source="test",
        category="research",
    )

    mock_episodic.record.assert_awaited_once()
    call_kwargs = mock_episodic.record.call_args.kwargs
    assert call_kwargs["source"] == "knowledge"
    assert "file_path" in call_kwargs.get("metadata", {})
