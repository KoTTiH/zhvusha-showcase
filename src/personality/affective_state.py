"""In-memory affective state tracker for Zhvusha.

Maintains a running estimate of Zhvusha's emotional state based on
enrichment results. The state is *locally scoped* (per the Anthropic
research): it decays toward baseline between interactions and does NOT
persist across process restarts.

Key mechanisms:
- **Counter-regulation**: when the user is highly aroused, Zhvusha
  dampens her own arousal (emotional thermostat).
- **Exponential decay**: emotions drift toward baseline (curiosity)
  with a half-life of 5 turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.personality.emotion_atlas import EMOTION_ATLAS, get_complement
from src.personality.protocols import AffectiveSnapshot, AffectiveStateProtocol

if TYPE_CHECKING:
    from src.memory import EnrichmentResult

# Zhvusha's resting state: curious, moderately engaged.
_BASELINE_SELF_EMOTION = "curiosity"
_BASELINE_SELF_VALENCE = 0.6
_BASELINE_SELF_AROUSAL = 0.6
_BASELINE_USER_EMOTION = "neutral"
_BASELINE_USER_VALENCE = 0.0
_BASELINE_USER_AROUSAL = 0.5

# Counter-regulation thresholds
_HIGH_AROUSAL_THRESHOLD = 0.7
_NEGATIVE_VALENCE_THRESHOLD = -0.3
_ACTIVE_AROUSAL_THRESHOLD = 0.5

# Decay half-life in turns
_DECAY_HALF_LIFE = 5.0

# Valence mapping for string → float
_VALENCE_MAP: dict[str, float] = {
    "positive": 0.5,
    "negative": -0.5,
    "neutral": 0.0,
}


def _lerp(baseline: float, current: float, factor: float) -> float:
    """Linear interpolation: baseline + (current - baseline) * factor."""
    return baseline + (current - baseline) * factor


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class AffectiveState:
    """Snapshot of Zhvusha's current emotional state."""

    # Zhvusha's own state
    self_emotion: str = _BASELINE_SELF_EMOTION
    self_valence: float = _BASELINE_SELF_VALENCE
    self_arousal: float = _BASELINE_SELF_AROUSAL

    # User's state (separate representation)
    user_emotion: str = _BASELINE_USER_EMOTION
    user_valence: float = _BASELINE_USER_VALENCE
    user_arousal: float = _BASELINE_USER_AROUSAL

    # Regulation
    regulation_active: bool = False
    regulation_target: str = ""

    # Decay
    turns_since_update: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class AffectiveStateManager(AffectiveStateProtocol):
    """Singleton manager for Zhvusha's affective state."""

    def __init__(self) -> None:
        self._state = AffectiveState()
        self._baseline = AffectiveState()
        self._is_at_baseline = True

    def get_state(self) -> AffectiveSnapshot:
        """Return frozen snapshot of current state.

        Snapshot mirrors :class:`AffectiveState` 1:1; callers cannot
        accidentally mutate manager state through the returned object.
        """
        s = self._state
        return AffectiveSnapshot(
            self_emotion=s.self_emotion,
            self_valence=s.self_valence,
            self_arousal=s.self_arousal,
            user_emotion=s.user_emotion,
            user_valence=s.user_valence,
            user_arousal=s.user_arousal,
            regulation_active=s.regulation_active,
            regulation_target=s.regulation_target,
            turns_since_update=s.turns_since_update,
            last_updated=s.last_updated,
        )

    def update_from_enrichment(self, result: EnrichmentResult) -> None:
        """Update state from a new enrichment result."""
        s = self._state

        # User side
        s.user_emotion = result.emotion
        s.user_valence = _VALENCE_MAP.get(result.valence, 0.0)
        s.user_arousal = _clamp(result.arousal, 0.0, 1.0)

        # Self side (from LLM's assessment of Zhvusha's reaction)
        s.self_emotion = result.self_emotion
        if result.self_emotion in EMOTION_ATLAS:
            concept = EMOTION_ATLAS[result.self_emotion]
            s.self_valence = concept.valence
        s.self_arousal = _clamp(result.self_arousal, 0.0, 1.0)

        # Reset decay
        s.turns_since_update = 0
        s.last_updated = datetime.now(tz=UTC)
        self._is_at_baseline = False

        # Apply counter-regulation
        self._apply_counter_regulation()

    def _apply_counter_regulation(self) -> None:
        """Dampen Zhvusha's arousal when user is highly aroused or negative."""
        s = self._state

        if s.user_arousal > _HIGH_AROUSAL_THRESHOLD:
            # User is intense → Zhvusha dampens
            comp = (
                get_complement(s.self_emotion)
                if s.self_emotion in EMOTION_ATLAS
                else "calm"
            )
            s.regulation_active = True
            s.regulation_target = comp
            s.self_arousal = _clamp(s.self_arousal * 0.6, 0.2, 0.9)
        elif (
            s.user_valence < _NEGATIVE_VALENCE_THRESHOLD
            and s.user_arousal > _ACTIVE_AROUSAL_THRESHOLD
        ):
            # User is negative + active → Zhvusha shifts to warmth
            s.regulation_active = True
            s.regulation_target = "warmth"
            s.self_arousal = _clamp(0.4, 0.2, 0.9)
        else:
            s.regulation_active = False
            s.regulation_target = ""

    def decay_if_stale(self) -> None:
        """Apply exponential decay toward baseline.

        Half-life is 5 turns: after 5 turns, state is 50% back to baseline.
        """
        turns = self._state.turns_since_update
        if turns <= 0 or self._is_at_baseline:
            return

        factor = 0.5 ** (turns / _DECAY_HALF_LIFE)

        s = self._state
        b = self._baseline
        s.self_valence = _clamp(
            _lerp(b.self_valence, s.self_valence, factor), -1.0, 1.0
        )
        s.self_arousal = _clamp(_lerp(b.self_arousal, s.self_arousal, factor), 0.0, 1.0)
        s.user_valence = _clamp(
            _lerp(b.user_valence, s.user_valence, factor), -1.0, 1.0
        )
        s.user_arousal = _clamp(_lerp(b.user_arousal, s.user_arousal, factor), 0.0, 1.0)

        # If close enough to baseline, snap back fully
        if (
            abs(s.self_valence - b.self_valence) < 0.1
            and abs(s.self_arousal - b.self_arousal) < 0.1
        ):
            s.self_emotion = b.self_emotion
            s.self_valence = b.self_valence
            s.self_arousal = b.self_arousal
            s.regulation_active = False
            s.regulation_target = ""
            self._is_at_baseline = True

    def get_prompt_context(self) -> str:
        """Return compact emotional context for the system prompt.

        Returns empty string at baseline (no noise).
        """
        if self._is_at_baseline:
            return ""

        # Apply decay before generating context, then advance counter
        self.decay_if_stale()
        self._state.turns_since_update += 1

        if self._is_at_baseline:
            return ""

        s = self._state
        self_ru = (
            EMOTION_ATLAS[s.self_emotion].name_ru
            if s.self_emotion in EMOTION_ATLAS
            else s.self_emotion
        )
        parts = [
            f"Моё состояние: {self_ru} (v={s.self_valence:+.1f} a={s.self_arousal:.1f})"
        ]

        if s.user_emotion != "neutral":
            parts.append(
                f"Никита: {s.user_emotion} (v={s.user_valence:+.1f} a={s.user_arousal:.1f})"
            )

        if s.regulation_active and s.regulation_target:
            target_ru = (
                EMOTION_ATLAS[s.regulation_target].name_ru
                if s.regulation_target in EMOTION_ATLAS
                else s.regulation_target
            )
            parts.append(f"Регуляция: сдвигаюсь к {target_ru}")

        return ". ".join(parts)

    def get_snapshot_line(self) -> str:
        """One-line summary for diary/emotional log."""
        s = self._state
        return (
            f"self={s.self_emotion}(v={s.self_valence:+.1f},a={s.self_arousal:.1f}) "
            f"user={s.user_emotion}(v={s.user_valence:+.1f},a={s.user_arousal:.1f})"
            + (f" reg→{s.regulation_target}" if s.regulation_active else "")
        )


_manager: AffectiveStateManager | None = None


def get_affective_state_manager() -> AffectiveStateManager:
    """Singleton accessor for AffectiveStateManager."""
    global _manager
    if _manager is None:
        _manager = AffectiveStateManager()
    return _manager
