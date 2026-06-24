"""Shared test fixtures for Phase 2+ tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def clean_repo_scoped_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-commit exports outer-repo Git vars; tests use nested tmp repos."""
    for key in (
        "GIT_INDEX_FILE",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_PREFIX",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_settings(tmp_path: Path) -> SimpleNamespace:
    """Settings-like object with all fields, using tmp_path for workspace."""
    return SimpleNamespace(
        bot_token="test_token",
        channel_id="@test_channel",
        admin_user_id=12345,
        google_api_key="",
        anthropic_api_key="",
        openrouter_api_key="",
        compare_main_tier="worker",
        compare_provider="",
        compare_model="",
        assistant_daily_message_limit=30,
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        kwork_login="",
        kwork_password="",
        kwork_phone_last="",
        kwork_poll_interval_seconds=300,
        kwork_min_budget=3000,
        kwork_max_offers=15,
        kwork_keywords="python,aiogram",
        claude_cli_path="",
        codex_cli_path="codex",
        code_agent_backend="codex_cli",
        code_agent_model="",
        workspace_path=str(tmp_path / "zhvusha-workspace"),
        project_path=str(tmp_path / "zhvusha-project"),
        git_max_commits=100,
        morning_session_model="gpt-5.5",
        morning_session_reasoning_effort="xhigh",
        morning_session_hour=8,
        morning_session_enabled=False,
        autonomous_self_coding_enabled=False,
        autonomous_self_coding_interval_seconds=21600,
        autonomous_self_coding_initial_delay_seconds=300,
        autonomous_self_coding_max_tier=3,
        chat_log_dir="",
        public_info_about_nikita="Test info",
        default_llm_tier="worker",
        worker_provider="codex_cli",
        analyst_provider="codex_cli",
        strategist_provider="codex_cli",
        vision_provider="gemini",
        worker_model="default",
        analyst_model="gpt-5.5",
        strategist_model="gpt-5.5",
        vision_model="gemini-2.5-flash-lite",
        worker_reasoning_effort="medium",
        analyst_reasoning_effort="high",
        strategist_reasoning_effort="xhigh",
        strategist_budget_daily_usd=1.00,
        enable_browser_use=False,
        # Phase 2 settings
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        recency_half_life_personal=48,
        recency_half_life_assistant=24,
        recency_half_life_social=12,
        reconsolidation_window_hours=6,
        system1_default_threshold=0.7,
        consolidation_top_n=300,
        core_md_max_lines=30,
        memory_index_max_lines=200,
        memory_index_max_kb=25,
        enrichment_tier="worker",
        enrichment_min_length=15,
        enrichment_max_concurrent=3,
        dream_extraction_tier="worker",
        # Phase 3 settings
        firefox_profile_path="",
        chrome_history_path="",
        youtube_takeout_path="",
        youtube_api_key="",
        youtube_scan_enabled=False,
        youtube_transcribe_top_n=3,
        telegram_api_id=0,
        telegram_api_hash="",
        telethon_session_path="~/.zhvusha_telethon.session",
        monitored_channel_ids="",
        channel_read_delay_seconds=1.5,
    )


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Minimal workspace directory structure for testing."""
    ws = tmp_path / "zhvusha-workspace"
    (ws / "personality").mkdir(parents=True)
    (ws / "personality" / "core.md").write_text(
        "# Who I Am\nI am Zhvusha.\n", encoding="utf-8"
    )
    (ws / "personality" / "genes.md").write_text(
        "# Genes\n| Gene | Value |\n|------|-------|\n| Curiosity | HIGH |\n",
        encoding="utf-8",
    )
    (ws / "memory" / "people").mkdir(parents=True)
    (ws / "diary").mkdir(parents=True)
    (ws / "inbox").mkdir(parents=True)
    return ws


@pytest.fixture
def mock_session_maker() -> MagicMock:
    """Mock SQLAlchemy async_sessionmaker with context manager support."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    maker = MagicMock(return_value=session)
    maker._mock_session = session  # expose for assertions
    return maker


@pytest.fixture
def mock_episodic() -> AsyncMock:
    """Mock EpisodicMemory with all public methods.

    Stateful: `record` increments a counter and stores the episode in
    `_episodes`; `update_enrichment` mutates the stored dict in place.
    This lets integration tests verify end-to-end enrichment without a
    live database. Access via `mock_episodic._episodes[episode_id]`.
    """
    ep = AsyncMock()
    episodes: dict[int, dict[str, Any]] = {}
    counter = {"next_id": 0}

    async def _record(**kwargs: Any) -> int:
        counter["next_id"] += 1
        ep_id = counter["next_id"]
        episodes[ep_id] = {
            "id": ep_id,
            "importance": kwargs.get("importance", 0.5),
            "valence": kwargs.get("valence", "neutral"),
            "confidence": kwargs.get("confidence", 0.5),
            "intent": None,
            "emotion": None,
            **kwargs,
        }
        return ep_id

    async def _update_enrichment(episode_id: int, result: Any) -> None:
        if episode_id not in episodes:
            return
        episodes[episode_id]["importance"] = result.importance
        episodes[episode_id]["valence"] = result.valence
        episodes[episode_id]["confidence"] = result.confidence
        episodes[episode_id]["intent"] = result.intent
        episodes[episode_id]["emotion"] = result.emotion

    ep.record = AsyncMock(side_effect=_record)
    ep.update_enrichment = AsyncMock(side_effect=_update_enrichment)
    ep.retrieve = AsyncMock(return_value=[])
    ep.retrieve_by_somatic_marker = AsyncMock(return_value=[])
    ep.complete_pattern = AsyncMock(return_value=None)
    ep.check_pattern_separation = AsyncMock(return_value=[])
    ep.get_unconsolidated = AsyncMock(return_value=[])
    ep.mark_consolidated = AsyncMock()
    ep.update_importance = AsyncMock()
    ep.update_valence = AsyncMock()
    ep._episodes = episodes  # expose for assertions
    return ep


FIXED_EMBEDDING = [0.1] * 384


@pytest.fixture
def mock_embedding_service(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch EmbeddingService.embed to return a fixed 384-dim vector."""
    from src.embeddings import EmbeddingService

    monkeypatch.setattr(
        EmbeddingService,
        "embed",
        classmethod(lambda cls, text: list(FIXED_EMBEDDING)),
    )
    monkeypatch.setattr(
        EmbeddingService,
        "embed_batch",
        classmethod(lambda cls, texts: [list(FIXED_EMBEDDING) for _ in texts]),
    )
    return list(FIXED_EMBEDDING)
