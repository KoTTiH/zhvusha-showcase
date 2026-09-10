from __future__ import annotations

from datetime import UTC, datetime

from src.daemon.signals import Signal
from src.memory.types import LearningSignal
from src.personality.protocols import AffectiveSnapshot, HomeostasisCorrection


def _snapshot(
    *,
    self_emotion: str = "curiosity",
    self_arousal: float = 0.85,
    regulation_active: bool = False,
) -> AffectiveSnapshot:
    return AffectiveSnapshot(
        self_emotion=self_emotion,
        self_valence=0.4,
        self_arousal=self_arousal,
        user_emotion="neutral",
        user_valence=0.0,
        user_arousal=0.3,
        regulation_active=regulation_active,
        regulation_target="",
        turns_since_update=1,
        last_updated=datetime(2026, 5, 14, tzinfo=UTC),
    )


def test_builder_turns_personality_desire_feedback_and_daemon_signals_into_intent() -> (
    None
):
    from src.agency.intent_builder import PersonalityDrivenIntentBuilder
    from src.agency.models import (
        AgencyActionKind,
        AgencyDataNeed,
        AgencyIntentKind,
        AgencyOutcomeKind,
    )

    feedback = LearningSignal(
        type="correction",
        statement="Не чинить архитектурные баги узкими if/else.",
        scope="work",
        confidence=0.94,
        apply_immediately=True,
        original_claim="Можно закрыть live баг phrase branch-ем.",
    )
    daemon_signal = Signal(
        source="daemon",
        signal_type="topic_cluster_ready",
        payload={"topic": "Telegram MCP social grants", "urgency": "high"},
    )

    intent = PersonalityDrivenIntentBuilder().build(
        event="Довести social agency до общего архитектурного контура",
        affective_snapshot=_snapshot(),
        homeostasis_corrections=(
            HomeostasisCorrection(
                gene="curiosity",
                direction="too_low",
                evidence="мало исследовательских действий",
                suggestion="вернуть исследование через runtime",
            ),
        ),
        desire_signals=("хочу лучше понимать, когда нужно спросить человека",),
        learning_signals=(feedback,),
        daemon_signals=(daemon_signal,),
        memory_evidence=("workspace/personality/dreams.md:12",),
    )

    assert intent.kind is AgencyIntentKind.SELF_COMPLEXIFICATION
    assert intent.source == "personality_driven_agency"
    assert intent.priority >= 70
    assert intent.why_personality_matters
    assert intent.drive_vector["curiosity"] > 0.7
    assert "feedback:correction" in intent.personality_drivers
    assert "daemon:topic_cluster_ready" in intent.personality_drivers
    assert AgencyDataNeed.HUMAN_OPINION in intent.data_needs
    assert AgencyOutcomeKind.SPEC in intent.expected_outcomes
    assert AgencyOutcomeKind.MEMORY_CANDIDATE in intent.expected_outcomes
    assert {action.kind for action in intent.candidate_actions} >= {
        AgencyActionKind.READ_WORKSPACE,
        AgencyActionKind.TELEGRAM_MCP_READ,
        AgencyActionKind.CREATE_SPEC,
        AgencyActionKind.STAGE_MEMORY,
    }
    assert any("ToolGateway" in item for item in intent.safety_constraints)


def test_same_event_with_different_personality_drivers_changes_priority_and_actions() -> (
    None
):
    from src.agency.intent_builder import PersonalityDrivenIntentBuilder
    from src.agency.models import AgencyActionKind

    builder = PersonalityDrivenIntentBuilder()

    curious = builder.build(
        event="Проверить архитектурный разрыв в agency",
        affective_snapshot=_snapshot(self_emotion="curiosity", self_arousal=0.95),
        desire_signals=("хочу разобраться глубже",),
    )
    regulated = builder.build(
        event="Проверить архитектурный разрыв в agency",
        affective_snapshot=_snapshot(
            self_emotion="overwhelm",
            self_arousal=0.25,
            regulation_active=True,
        ),
        desire_signals=("нужно не сжечь лимиты и сначала подготовить черновик",),
    )

    assert curious.priority > regulated.priority
    assert AgencyActionKind.WEB_RESEARCH in {
        action.kind for action in curious.candidate_actions
    }
    assert AgencyActionKind.DRAFT_MESSAGE in {
        action.kind for action in regulated.candidate_actions
    }
    assert any("low_arousal" in item for item in regulated.safety_constraints)
