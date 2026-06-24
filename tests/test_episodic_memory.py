"""Tests for EpisodicMemory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.episodic import _SOCIAL_RATE_LIMIT, EpisodicMemory


def _make_episode(**overrides):
    """Create a minimal Episode-like object for testing.

    Mirrors all 21 columns of the ORM Episode model so that
    EpisodicMemory._orm_to_domain can safely read every attribute when
    building the frozen domain Episode returned by public methods.
    """
    defaults = {
        "id": 1,
        "content": "test content",
        "summary": None,
        "user_id": 12345,
        "chat_type": "personal",
        "role": "user",
        "importance": 0.5,
        "valence": "neutral",
        "confidence": 0.5,
        "embedding": [0.1] * 384,
        "access_count": 0,
        "timestamp": datetime.now(tz=UTC),
        "last_accessed": None,
        "consolidated": False,
        "consolidation_result": None,
        "source": "chat",
        "enrichment_status": "pending",
        "intent": None,
        "emotion": None,
        "embedding_version": 1,
        "metadata_json": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_embed():
    return patch(
        "src.memory.episodic.EmbeddingService.embed",
        return_value=[0.5] * 384,
    )


def _patch_cosine(val=0.8):
    return patch(
        "src.memory.episodic.EmbeddingService.cosine_similarity",
        return_value=val,
    )


# --- record() tests ---


async def test_record_personal_full_content(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    with _patch_embed() as mock_embed:
        await ep.record(
            content="важный разговор с никитой",
            user_id=12345,
            chat_type="personal",
            role="user",
        )

    # Should have called embed for personal mode
    mock_embed.assert_called_once()
    # Should have added and committed
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


async def test_record_personal_continues_without_embedding_on_model_failure(
    mock_session_maker,
):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    with patch(
        "src.memory.episodic.EmbeddingService.embed_async",
        side_effect=TypeError("Pooling.__init__() missing word_embedding_dimension"),
    ):
        await ep.record(
            content="важный разговор с никитой",
            user_id=12345,
            chat_type="personal",
            role="user",
        )

    added = session.add.call_args[0][0]
    assert added.embedding is None
    session.commit.assert_awaited_once()


async def test_record_assistant_lowered_importance(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)

    with _patch_embed():
        # Explicit importance should be preserved
        await ep.record(
            content="клиент спрашивает про бота",
            user_id=999,
            chat_type="assistant",
            role="user",
            importance=0.3,
        )

    added = mock_session_maker._mock_session.add.call_args[0][0]
    assert added.importance == 0.3


async def test_record_social_metadata_only(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)

    with _patch_embed() as mock_embed:
        await ep.record(
            content="a" * 200,  # Long message
            user_id=777,
            chat_type="social",
            role="user",
        )

    # Social: should NOT generate embedding
    mock_embed.assert_not_called()
    # Content should be truncated to 100 chars
    added = mock_session_maker._mock_session.add.call_args[0][0]
    assert len(added.content) == 100
    assert added.importance == 0.1


async def test_record_accepts_enrichment_params(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)

    with _patch_embed() as mock_embed:
        await ep.record(
            content="тест",
            user_id=12345,
            chat_type="personal",
            role="user",
            person_name="Никита",
            significance="inner_circle",
            domain="kwork",
        )

    # Tier 1 enrichment should include metadata in embed text
    call_args = mock_embed.call_args[0][0]
    assert "person:Никита" in call_args
    assert "significance:inner_circle" in call_args
    assert "domain:kwork" in call_args


async def test_social_rate_limit_blocks_after_max(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)

    with _patch_embed():
        # Record up to limit
        for _ in range(_SOCIAL_RATE_LIMIT):
            result = await ep.record(
                content="msg", user_id=777, chat_type="social", role="user"
            )
            assert result != -1

        # Next should be blocked
        result = await ep.record(
            content="msg", user_id=777, chat_type="social", role="user"
        )
        assert result == -1


# --- retrieve() tests ---


async def test_retrieve_returns_scored_episodes(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episodes = [
        _make_episode(id=1, importance=0.8, access_count=5),
        _make_episode(id=2, importance=0.3, access_count=1),
    ]

    # Mock the execute result
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = episodes
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed(), _patch_cosine(0.8):
        results = await ep.retrieve("test query", limit=2)

    assert len(results) <= 2
    # Higher importance should score higher
    if len(results) == 2:
        assert results[0].importance >= results[1].importance


async def test_retrieve_increments_access_count(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episodes = [_make_episode(id=1)]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = episodes
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed(), _patch_cosine(0.8):
        await ep.retrieve("test", limit=1)

    # Should have called execute twice: SELECT + UPDATE
    assert session.execute.await_count == 2
    session.commit.assert_awaited()


async def test_retrieve_empty_when_no_candidates(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed():
        results = await ep.retrieve("test")

    assert results == []


# --- somatic marker tests ---


async def test_somatic_marker_returns_valence(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episodes = [
        _make_episode(id=1, valence="positive", confidence=0.9),
    ]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = episodes
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed(), _patch_cosine(0.85):
        results = await ep.retrieve_by_somatic_marker("test")

    assert len(results) == 1
    episode, similarity = results[0]
    assert episode.valence == "positive"
    assert similarity > 0.5


# --- pattern completion ---


async def test_complete_pattern_returns_match(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episode = _make_episode(id=1, content="full context about client bot project")
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = episode
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed(), _patch_cosine(0.75):
        result = await ep.complete_pattern("that client bot")

    assert result is not None
    assert result.content == "full context about client bot project"


async def test_complete_pattern_returns_none_below_threshold(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episode = _make_episode(id=1)
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = episode
    session.execute = AsyncMock(return_value=mock_result)

    with _patch_embed(), _patch_cosine(0.3):
        result = await ep.complete_pattern("unrelated query")

    assert result is None


# --- unconsolidated ---


async def test_get_unconsolidated(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episodes = [_make_episode(consolidated=False)]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = episodes
    session.execute = AsyncMock(return_value=mock_result)

    results = await ep.get_unconsolidated()
    assert len(results) == 1


async def test_mark_consolidated(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    await ep.mark_consolidated([1, 2, 3], result="processed")
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


# --- update_importance with reconsolidation window ---


async def test_update_importance_within_window(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episode = _make_episode(
        id=1,
        last_accessed=datetime.now(tz=UTC) - timedelta(hours=1),
        importance=0.5,
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = episode
    session.execute = AsyncMock(return_value=mock_result)

    await ep.update_importance(1, 0.9, reconsolidation_window_hours=6)
    # Within window — should update
    assert episode.importance == 0.9


async def test_update_importance_outside_window(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    episode = _make_episode(
        id=1,
        last_accessed=datetime.now(tz=UTC) - timedelta(hours=12),
        importance=0.5,
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = episode
    session.execute = AsyncMock(return_value=mock_result)

    await ep.update_importance(1, 0.9, reconsolidation_window_hours=6)
    # Outside window — should NOT update
    assert episode.importance == 0.5


# --- update_valence ---


async def test_update_valence(mock_session_maker):
    ep = EpisodicMemory(mock_session_maker, admin_user_id=12345)
    session = mock_session_maker._mock_session

    await ep.update_valence(1, "positive", 0.9)
    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


# --- Tier 1 enrichment ---


def test_tier1_embed_text_includes_metadata():
    text = EpisodicMemory._build_tier1_embed_text(
        content="тестовое сообщение",
        source="chat",
        chat_type="personal",
        role="user",
        person_name="Никита",
        significance="inner_circle",
        domain="chat",
        importance=0.7,
    )
    assert "person:Никита" in text
    assert "significance:inner_circle" in text
    assert "domain:chat" in text
    assert "importance:0.7" in text
    assert "тестовое сообщение" in text


def test_tier1_detects_question():
    text = EpisodicMemory._build_tier1_embed_text(
        content="как дела?",
        source="chat",
        chat_type="personal",
        role="user",
        person_name="test",
        significance="stranger",
        domain="chat",
        importance=0.5,
    )
    assert "type:question" in text


# --- embedding_version default ---


def test_episode_defaults():
    """Episode enrichment fields have correct defaults."""
    from src.memory.database import Episode as EpisodeModel

    table = EpisodeModel.__table__
    cols = {c.name: c for c in table.columns}
    assert cols["enrichment_status"].default.arg == "pending"
    assert cols["embedding_version"].default.arg == 1
