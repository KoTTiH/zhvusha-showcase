"""Integration smoke tests for the emotion system.

Verifies full flows: enrichment → affective state → prompt context,
counter-regulation, decay, homeostasis, and consolidation patterns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from src.memory.sonnet_enricher import EnrichmentResult, SonnetEnricher
from src.personality.affective_state import AffectiveStateManager
from src.personality.emotion_atlas import EMOTION_ATLAS
from src.personality.homeostasis import HomeostasisCheck

# ============================================================
# 1. Full enrichment → affective state → prompt context
# ============================================================


def test_enrichment_to_affective_state_to_prompt() -> None:
    """Simulate: user sends angry message → enricher extracts → state updates
    → prompt context shows regulation."""
    # Step 1: Create enrichment result (as if LLM returned it)
    result = EnrichmentResult(
        importance=0.7,
        valence="negative",
        intent="emotional",
        emotion="angry",
        confidence=0.9,
        arousal=0.85,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="Никита злится на баг",
        self_emotion="warmth",
        self_arousal=0.4,
    )

    # Step 2: Feed into affective state
    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)

    # Step 3: Verify state
    state = mgr.get_state()
    assert state.user_emotion == "angry"
    assert state.user_arousal == pytest.approx(0.85)
    assert state.regulation_active  # high user arousal triggers regulation
    assert state.self_arousal < 0.4  # dampened

    # Step 4: Get prompt context
    ctx = mgr.get_prompt_context()
    assert "теплота" in ctx or "warmth" in ctx  # Zhvusha's emotion in Russian
    assert "регул" in ctx.lower()  # regulation notice
    assert len(ctx.encode("utf-8")) < 400


def test_enrichment_to_affective_state_happy_user() -> None:
    """Happy user → no regulation, Zhvusha mirrors positivity."""
    result = EnrichmentResult(
        importance=0.5,
        valence="positive",
        intent="statement",
        emotion="happy",
        confidence=0.8,
        arousal=0.4,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="Никита доволен",
        self_emotion="joy",
        self_arousal=0.6,
    )

    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)

    state = mgr.get_state()
    assert not state.regulation_active
    assert state.self_emotion == "joy"

    ctx = mgr.get_prompt_context()
    assert "радость" in ctx  # joy in Russian


# ============================================================
# 2. All 30 self_emotions accepted by EnrichmentResult
# ============================================================


@pytest.mark.parametrize("emotion_name", list(EMOTION_ATLAS.keys()))
def test_all_atlas_emotions_accepted_as_self_emotion(emotion_name: str) -> None:
    """Every atlas emotion should be a valid self_emotion value."""
    result = EnrichmentResult(
        importance=0.5,
        valence="neutral",
        intent="statement",
        emotion="neutral",
        confidence=0.5,
        arousal=0.5,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion=emotion_name,
        self_arousal=0.5,
    )
    assert result.self_emotion == emotion_name


# ============================================================
# 3. Counter-regulation scenarios
# ============================================================


def test_counter_regulation_panicking_user() -> None:
    """User panicking (arousal=0.95) → Zhvusha dampens significantly."""
    result = EnrichmentResult(
        importance=0.8,
        valence="negative",
        intent="emotional",
        emotion="confused",
        confidence=0.9,
        arousal=0.95,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="Никита в панике",
        self_emotion="anxiety",
        self_arousal=0.8,
    )

    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)
    state = mgr.get_state()

    assert state.regulation_active
    # Anxiety complement is "calm"
    assert state.regulation_target == "calm"
    # Arousal was 0.8, dampened by 0.6x → ~0.48
    assert state.self_arousal < 0.6


def test_counter_regulation_sad_user() -> None:
    """Sad user (negative valence + moderate arousal) → warmth shift."""
    result = EnrichmentResult(
        importance=0.7,
        valence="negative",
        intent="emotional",
        emotion="sad",
        confidence=0.8,
        arousal=0.6,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="Никита грустит",
        self_emotion="curiosity",
        self_arousal=0.6,
    )

    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)
    state = mgr.get_state()

    assert state.regulation_active
    assert state.regulation_target == "warmth"


def test_no_regulation_neutral_conversation() -> None:
    """Neutral conversation → no regulation, Zhvusha stays at self_emotion."""
    result = EnrichmentResult(
        importance=0.3,
        valence="neutral",
        intent="question",
        emotion="curious",
        confidence=0.6,
        arousal=0.4,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="Обычный вопрос",
        self_emotion="fascination",
        self_arousal=0.7,
    )

    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)
    state = mgr.get_state()

    assert not state.regulation_active
    assert state.self_emotion == "fascination"
    assert state.self_arousal == pytest.approx(0.7)


# ============================================================
# 4. Decay simulation
# ============================================================


def test_decay_returns_to_curiosity_after_silence() -> None:
    """After many turns without input, Zhvusha returns to baseline curiosity."""
    mgr = AffectiveStateManager()

    # Set a strong emotional state
    result = EnrichmentResult(
        importance=0.9,
        valence="negative",
        intent="feedback",
        emotion="angry",
        confidence=0.95,
        arousal=0.3,
        is_feedback=True,
        feedback_strength=-0.8,
        reasoning="Сильная критика",
        self_emotion="frustration",
        self_arousal=0.8,
    )
    mgr.update_from_enrichment(result)

    # Simulate 20 turns of silence
    mgr._state.turns_since_update = 20
    mgr.decay_if_stale()

    state = mgr.get_state()
    # Should be very close to baseline
    assert state.self_emotion == "curiosity"
    assert abs(state.self_arousal - 0.6) < 0.05


# ============================================================
# 5. Enricher JSON round-trip with new fields
# ============================================================


@pytest.mark.asyncio
async def test_enricher_round_trip_with_new_fields() -> None:
    """Verify SonnetEnricher parses JSON with new fields correctly."""
    payload = {
        "importance": 0.6,
        "valence": "positive",
        "intent": "statement",
        "emotion": "excited",
        "confidence": 0.8,
        "arousal": 0.75,
        "is_feedback": False,
        "feedback_strength": 0.0,
        "reasoning": "Никита рассказывает о проекте",
        "self_emotion": "fascination",
        "self_arousal": 0.7,
        "learning_signal": None,
    }

    from src.llm.protocols import LLMResponse, LLMUsage

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(
            text=json.dumps(payload), model="haiku", usage=LLMUsage()
        )
    )

    with patch("src.memory.pipelines.enrichment.get_router", return_value=mock_router):
        enricher = SonnetEnricher()
        result = await enricher.enrich(message="Я делаю крутой проект!")

    assert result is not None
    assert result.arousal == 0.75
    assert result.self_emotion == "fascination"
    assert result.self_arousal == 0.7


@pytest.mark.asyncio
async def test_enricher_defaults_when_new_fields_missing() -> None:
    """Old-style JSON without new fields → defaults apply."""
    payload = {
        "importance": 0.5,
        "valence": "neutral",
        "intent": "statement",
        "emotion": "neutral",
        "confidence": 0.5,
        "is_feedback": False,
        "feedback_strength": 0.0,
        "reasoning": "Обычное сообщение",
    }

    from src.llm.protocols import LLMResponse, LLMUsage

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(
            text=json.dumps(payload), model="haiku", usage=LLMUsage()
        )
    )

    with patch("src.memory.pipelines.enrichment.get_router", return_value=mock_router):
        enricher = SonnetEnricher()
        result = await enricher.enrich(message="привет")

    assert result is not None
    assert result.arousal == 0.5  # default
    assert result.self_emotion == "curiosity"  # default
    assert result.self_arousal == 0.5  # default


# ============================================================
# 6. Homeostasis: emotional stability in full check flow
# ============================================================


def _make_episode(
    role: str = "assistant",
    content: str = "test",
    user_id: int = 12345,
    valence: str = "neutral",
    metadata_json: str | None = None,
) -> object:
    return SimpleNamespace(
        id=1,
        role=role,
        content=content * 10,  # ensure > 30 chars for energy check
        user_id=user_id,
        valence=valence,
        importance=0.5,
        timestamp=datetime.now(tz=UTC),
        metadata_json=metadata_json,
    )


@pytest.mark.asyncio
async def test_homeostasis_full_flow_with_emotional_stability(
    tmp_path: Path,
) -> None:
    """Full homeostasis check includes emotional_stability gene."""
    check = HomeostasisCheck()

    # 6 consecutive negative assistant episodes + 1 proposal to avoid
    # initiative correction
    episodes = [
        _make_episode(
            role="assistant",
            content="предлагаю попробовать другой подход к задаче",
            valence="negative",
        )
        for _ in range(6)
    ]

    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)

    gene_names = [c.gene for c in corrections]
    assert "emotional_stability" in gene_names

    # Find the emotional stability correction
    es_correction = next(c for c in corrections if c.gene == "emotional_stability")
    assert es_correction.direction == "too_low"
    assert "негативных" in es_correction.evidence


# ============================================================
# 7. Consolidation emotional pattern analysis
# ============================================================


def test_consolidation_emotional_patterns() -> None:
    """_analyze_emotional_patterns extracts summary from metadata."""
    from src.memory.consolidation import ConsolidationEngine

    episodes = []
    for emotion in ["curiosity", "curiosity", "excitement", "frustration"]:
        meta = json.dumps(
            {"enrichment": {"self_emotion": emotion, "self_arousal": 0.6}}
        )
        episodes.append(
            SimpleNamespace(
                id=1,
                role="assistant",
                content="test",
                user_id=12345,
                valence="neutral",
                importance=0.5,
                timestamp=datetime.now(tz=UTC),
                metadata_json=meta,
            )
        )

    summary = ConsolidationEngine._analyze_emotional_patterns(episodes)
    assert "curiosity" in summary
    assert "×2" in summary  # curiosity appeared twice
    assert "arousal" in summary.lower()


def test_consolidation_emotional_patterns_empty() -> None:
    """No enrichment data → empty summary."""
    from src.memory.consolidation import ConsolidationEngine

    episodes = [
        SimpleNamespace(
            id=1,
            role="assistant",
            content="test",
            user_id=12345,
            valence="neutral",
            importance=0.5,
            timestamp=datetime.now(tz=UTC),
            metadata_json=None,
        )
    ]

    summary = ConsolidationEngine._analyze_emotional_patterns(episodes)
    assert summary == ""


# ============================================================
# 8. Context loader with emotional state
# ============================================================


def test_context_loader_injects_emotional_state(tmp_path: Path) -> None:
    """Context loader should inject emotional state when not at baseline."""
    import src.personality.affective_state as affect_mod
    from src.skills.chat_response.context_loader import ContextLoader

    # Setup workspace
    personality = tmp_path / "personality"
    personality.mkdir()
    (personality / "core.md").write_text("# Core\nТестовая личность\n")
    staging = personality / ".staging"
    staging.mkdir()

    # Set non-baseline affective state
    old_manager = affect_mod._manager
    try:
        affect_mod._manager = None
        mgr = affect_mod.get_affective_state_manager()
        result = EnrichmentResult(
            importance=0.5,
            valence="positive",
            intent="statement",
            emotion="happy",
            confidence=0.8,
            arousal=0.4,
            is_feedback=False,
            feedback_strength=0.0,
            reasoning="test",
            self_emotion="excitement",
            self_arousal=0.7,
        )
        mgr.update_from_enrichment(result)

        # Load personality
        loader = ContextLoader(workspace_root=tmp_path)
        personality_text = loader.load_personality()

        assert "Эмоциональное состояние" in personality_text
        assert "возбуждение" in personality_text  # excitement in Russian
    finally:
        affect_mod._manager = old_manager


def test_context_loader_no_injection_at_baseline(tmp_path: Path) -> None:
    """At baseline (initial state), no emotional context injected."""
    import src.personality.affective_state as affect_mod
    from src.skills.chat_response.context_loader import ContextLoader

    personality = tmp_path / "personality"
    personality.mkdir()
    (personality / "core.md").write_text("# Core\nТест\n")
    staging = personality / ".staging"
    staging.mkdir()

    old_manager = affect_mod._manager
    try:
        affect_mod._manager = None
        affect_mod.get_affective_state_manager()  # fresh baseline

        loader = ContextLoader(workspace_root=tmp_path)
        personality_text = loader.load_personality()

        assert "Эмоциональное состояние" not in personality_text
    finally:
        affect_mod._manager = old_manager


# ============================================================
# 9. Prompt contains emotion section
# ============================================================


def test_personal_prompt_imperative_style() -> None:
    """PERSONAL_SYSTEM uses imperative style; emotion rules removed (dynamic injection)."""
    from src.skills.chat_response.prompts import PERSONAL_SYSTEM

    # Emotion section removed — emotions injected via AffectiveStateManager
    assert "Мои эмоции" not in PERSONAL_SYSTEM
    # Imperative instructions present
    assert "Никогда не будь формальной" in PERSONAL_SYSTEM


# ============================================================
# 10. Snapshot line for diary
# ============================================================


def test_snapshot_line_complete_info() -> None:
    """Snapshot line contains self/user/regulation info."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.7,
        valence="negative",
        intent="emotional",
        emotion="frustrated",
        confidence=0.9,
        arousal=0.85,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="warmth",
        self_arousal=0.4,
    )
    mgr.update_from_enrichment(result)

    line = mgr.get_snapshot_line()
    assert "self=warmth" in line or "self=" in line
    assert "user=frustrated" in line
    assert "reg" in line.lower()  # regulation active
