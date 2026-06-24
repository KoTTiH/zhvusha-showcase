"""Tests for ImportanceScorer."""

from types import SimpleNamespace
from unittest.mock import patch

from src.memory.importance import ImportanceScorer


def _make_episode(embedding=None, **kwargs):
    """Create a minimal episode-like object."""
    defaults = {
        "id": 1,
        "content": "test",
        "importance": 0.5,
        "valence": "neutral",
        "embedding": embedding,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _patch_embed(return_value=None):
    """Patch EmbeddingService.embed to return a fixed vector."""
    vec = return_value or [0.5] * 384
    return patch("src.memory.importance.EmbeddingService.embed", return_value=vec)


def _patch_cosine(return_value=0.5):
    """Patch EmbeddingService.cosine_similarity."""
    return patch(
        "src.memory.importance.EmbeddingService.cosine_similarity",
        return_value=return_value,
    )


async def test_admin_feedback_gets_boost():
    scorer = ImportanceScorer()
    # "хорошо" is evaluative language from admin
    with _patch_embed():
        score = await scorer.score(
            content="хорошо сделала",
            user_id=12345,
            is_admin=True,
            chat_type="personal",
            recent_episodes=[],
        )
    # base 0.5 + admin boost 0.3 = 0.8
    assert score >= 0.8


async def test_novelty_gets_boost():
    scorer = ImportanceScorer()
    recent = [_make_episode(embedding=[0.1] * 384)]

    with _patch_embed(), _patch_cosine(return_value=0.1):
        # Low cosine = high surprise (1.0 - 0.1 = 0.9 > 0.7)
        score = await scorer.score(
            content="completely new topic",
            user_id=1,
            is_admin=False,
            chat_type="personal",
            recent_episodes=recent,
        )
    # base 0.5 + novelty 0.2 = 0.7
    assert score >= 0.7


async def test_explicit_remember_returns_max():
    scorer = ImportanceScorer()
    score = await scorer.score(
        content="запомни это навсегда",
        user_id=1,
        is_admin=False,
        chat_type="personal",
        recent_episodes=[],
    )
    assert score == 1.0


async def test_routine_message_gets_low_score():
    scorer = ImportanceScorer()
    recent = [_make_episode(embedding=[0.5] * 384)]

    with _patch_embed(), _patch_cosine(return_value=0.9):
        # High cosine = low surprise (1.0 - 0.9 = 0.1 < 0.7) → no novelty boost
        score = await scorer.score(
            content="привет как дела",
            user_id=1,
            is_admin=False,
            chat_type="personal",
            recent_episodes=recent,
        )
    # base 0.5 only, no boosts
    assert score == 0.5


async def test_angry_message_gets_penalty():
    scorer = ImportanceScorer()
    score = await scorer.score(
        content="БЛЯТЬ КАКОГО ХЕРА ЭТО НЕ РАБОТАЕТ!!!",
        user_id=12345,
        is_admin=True,
        chat_type="personal",
        recent_episodes=[],
    )
    # base 0.5 + possible admin boost, but emotional penalty
    # Penalty: caps > 3 (-0.1) + profanity (-0.1) + !!! (-0.05) → capped at -0.15
    assert score < 0.5


async def test_social_mode_base_importance():
    scorer = ImportanceScorer()
    score = await scorer.score(
        content="обычное сообщение в чате",
        user_id=999,
        is_admin=False,
        chat_type="social",
        recent_episodes=[],
    )
    assert score == 0.1
