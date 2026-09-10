"""Tests for DecisionEngine."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.core.decision import DecisionEngine
from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="sonnet", usage=LLMUsage())


def _make_context(mode="personal", **kwargs):
    defaults = {
        "user_id": 12345,
        "message": "test",
        "mode": mode,
        "metadata": {},
        "bot": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_episode(valence="positive", confidence=0.9, **kwargs):
    defaults = {
        "id": 1,
        "content": "similar past experience",
        "valence": valence,
        "confidence": confidence,
        "embedding": [0.5] * 384,
        "importance": 0.7,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _setup_engine():
    episodic = AsyncMock()
    personality = MagicMock()
    personality.get_personality_tree_summary.return_value = "I am Zhvusha."
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_llm_resp("LLM response"))

    engine = DecisionEngine(episodic, personality, llm)
    return engine, episodic, llm


# --- System 1 tests ---


async def test_system1_finds_similar_experience():
    engine, episodic, _ = _setup_engine()
    episode = _make_episode(valence="positive", confidence=0.9)
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context()
    decision = await engine.decide("привет, как дела?", context)

    assert decision.system == "system1"
    assert decision.domain == "chat"
    assert decision.confidence == 0.9


async def test_system1_no_match_falls_to_system2():
    engine, episodic, llm = _setup_engine()
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[])

    context = _make_context()
    decision = await engine.decide("что-то совсем новое", context)

    assert decision.system == "system2"
    # System 2 now calls LLM twice: depth classification + response
    assert llm.generate.await_count == 2


async def test_system1_negative_valence_suggests_avoidance():
    engine, episodic, _ = _setup_engine()
    episode = _make_episode(valence="negative", confidence=0.9)
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context()
    decision = await engine.decide("test", context)

    assert decision.system == "system1"
    assert decision.result.suggested_approach == "avoid"


async def test_system1_below_threshold_escalates():
    engine, episodic, _llm = _setup_engine()
    episode = _make_episode(confidence=0.3)  # Below default 0.7
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context()
    decision = await engine.decide("test", context)

    assert decision.system == "system2"


# --- Domain-based restrictions ---


async def test_system1_full_auto_in_chat_domain():
    engine, episodic, _ = _setup_engine()
    episode = _make_episode(confidence=0.9)
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context(mode="personal")  # → domain "chat"
    decision = await engine.decide("обычный вопрос", context)
    assert decision.system == "system1"


async def test_system1_suggest_only_in_kwork_domain():
    engine, episodic, llm = _setup_engine()
    episode = _make_episode(confidence=0.9)
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context(metadata={"source": "kwork"})
    decision = await engine.decide("новый заказ на кворке", context)

    # System 1 found match but kwork is suggest-only → System 2
    assert decision.system == "system2"
    # System 2 now calls LLM twice: depth classification + response
    assert llm.generate.await_count == 2


# --- System 2 with intuition ---


async def test_system2_includes_system1_intuition():
    engine, episodic, llm = _setup_engine()
    episode = _make_episode(confidence=0.9)
    episodic.retrieve_by_somatic_marker = AsyncMock(return_value=[(episode, 0.85)])

    context = _make_context(metadata={"source": "kwork"})
    decision = await engine.decide("заказ", context)

    assert decision.system == "system2"
    assert decision.result.system1_intuition is not None
    # LLM prompt should include intuition (second call is the response)
    request = llm.generate.call_args.args[0]
    assert "интуиция" in request.prompt.lower()
    assert "Непереписываемая личность" in request.system


# --- record_outcome ---


async def test_record_outcome_updates_valence():
    engine, episodic, _ = _setup_engine()

    await engine.record_outcome(1, "positive", "chat")
    episodic.update_valence.assert_awaited_once_with(1, "positive", 0.8)
