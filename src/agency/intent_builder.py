"""Personality-driven AgencyIntent builder.

This module consumes public signal contracts from personality, memory and daemon
surfaces. It does not call tools and does not approve actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.agency.models import (
    AgencyAction,
    AgencyActionKind,
    AgencyDataNeed,
    AgencyIntent,
    AgencyIntentKind,
    AgencyOutcomeKind,
    SocialPermissionScope,
)

if TYPE_CHECKING:
    from src.daemon.signals import Signal
    from src.memory.protocols import LearningSignal
    from src.personality.protocols import AffectiveSnapshot, HomeostasisCorrection

_CURIOUS_EMOTIONS = frozenset(
    {
        "curiosity",
        "wonder",
        "fascination",
        "excitement",
        "delight",
        "confidence",
    }
)
_SOCIAL_EMOTIONS = frozenset({"warmth", "tenderness", "gratitude", "playfulness"})
_CAUTION_EMOTIONS = frozenset(
    {"overwhelm", "anxiety", "nervousness", "frustration", "irritation"}
)
_SOCIAL_TERMS = frozenset(
    {
        "человек",
        "человечес",
        "мнение",
        "социал",
        "чат",
        "групп",
        "форум",
        "reply",
        "social",
        "opinion",
    }
)
_BUDGET_TERMS = frozenset(
    {"лимит", "budget", "дорог", "дешев", "сжечь", "cost", "token"}
)


@dataclass
class _IntentBuildState:
    drive_vector: dict[str, float] = field(
        default_factory=lambda: {"self_complexification": 0.6}
    )
    drivers: list[str] = field(default_factory=list)
    data_needs: set[AgencyDataNeed] = field(
        default_factory=lambda: {AgencyDataNeed.FACTS}
    )
    outcomes: set[AgencyOutcomeKind] = field(
        default_factory=lambda: {AgencyOutcomeKind.CONTEXT_CAPSULE}
    )
    actions: list[AgencyAction] = field(
        default_factory=lambda: [
            AgencyAction(
                kind=AgencyActionKind.READ_WORKSPACE,
                capability="read_workspace",
                description="Собрать локальный контекст перед решением.",
            )
        ]
    )
    safety_constraints: list[str] = field(
        default_factory=lambda: [
            "Read-only auto only; side effects require policy and approval gates.",
            "ToolGateway must physically deny send/publish/restart/env side effects.",
            "Tier 3 core/personality/safety/runtime/dispatcher changes remain pending for Никита.",
        ]
    )
    evidence: list[str] = field(default_factory=list)
    priority: int = 50


class PersonalityDrivenIntentBuilder:
    """Build AgencyIntent from Жвушины internal/body signals."""

    def build(
        self,
        *,
        event: str,
        affective_snapshot: AffectiveSnapshot | None = None,
        homeostasis_corrections: tuple[HomeostasisCorrection, ...] = (),
        desire_signals: tuple[str, ...] = (),
        learning_signals: tuple[LearningSignal, ...] = (),
        daemon_signals: tuple[Signal, ...] = (),
        memory_evidence: tuple[str, ...] = (),
    ) -> AgencyIntent:
        """Create one bounded intent without executing actions."""

        goal = _clean_non_blank(event, field_name="event")
        state = _IntentBuildState(evidence=list(memory_evidence))

        priority = 50
        if affective_snapshot is not None:
            priority += self._apply_affect(
                affective_snapshot,
                state=state,
            )

        priority += self._apply_homeostasis(homeostasis_corrections, state=state)

        if desire_signals:
            priority += self._apply_desires(
                desire_signals,
                state=state,
            )

        priority += self._apply_learning_signals(learning_signals, state=state)
        priority += self._apply_daemon_signals(daemon_signals, state=state)
        state.priority = priority
        _finalize_state(state)
        return _intent_from_state(goal=goal, state=state)

    def _apply_affect(
        self,
        snapshot: AffectiveSnapshot,
        *,
        state: _IntentBuildState,
    ) -> int:
        emotion = snapshot.self_emotion
        state.drivers.append(f"affect:{emotion}")
        priority_delta = 0
        if emotion in _CURIOUS_EMOTIONS:
            state.drive_vector["curiosity"] = max(
                state.drive_vector.get("curiosity", 0.0),
                max(0.55, snapshot.self_arousal),
            )
            _append_action(
                state.actions,
                AgencyAction(
                    kind=AgencyActionKind.WEB_RESEARCH,
                    capability="web_search_sources",
                    description="Любопытство требует evidence-backed read-only research.",
                ),
            )
            priority_delta += int(snapshot.self_arousal * 24)
        if emotion in _SOCIAL_EMOTIONS:
            state.drive_vector["social_calibration"] = max(
                state.drive_vector.get("social_calibration", 0.0),
                max(0.55, snapshot.self_arousal),
            )
            priority_delta += 10
        if emotion in _CAUTION_EMOTIONS or snapshot.regulation_active:
            state.drive_vector["caution"] = max(
                state.drive_vector.get("caution", 0.0),
                0.75,
            )
            state.safety_constraints.append(
                "low_arousal_or_regulation_active: prefer draft/read-only before action."
            )
            _append_action(
                state.actions,
                AgencyAction(
                    kind=AgencyActionKind.DRAFT_MESSAGE,
                    capability="agency_social_permission_request",
                    description="Сначала подготовить черновик решения для Жвушиного loop.",
                ),
            )
            priority_delta -= 25
        if snapshot.self_arousal < 0.35:
            state.safety_constraints.append(
                "low_arousal_budget_guard: keep effort bounded and avoid live loops."
            )
            priority_delta -= 12
        return priority_delta

    def _apply_homeostasis(
        self,
        corrections: tuple[HomeostasisCorrection, ...],
        *,
        state: _IntentBuildState,
    ) -> int:
        priority_delta = 0
        for correction in corrections:
            state.drivers.append(
                f"homeostasis:{correction.gene}:{correction.direction}"
            )
            state.evidence.append(f"homeostasis:{correction.evidence}")
            state.outcomes.add(AgencyOutcomeKind.SPEC)
            _append_action(
                state.actions,
                AgencyAction(
                    kind=AgencyActionKind.CREATE_SPEC,
                    capability="request_tier3_specs_for_nikita_approval",
                    description=correction.suggestion,
                ),
            )
            if correction.direction == "too_low":
                state.drive_vector[correction.gene] = max(
                    state.drive_vector.get(correction.gene, 0.0),
                    0.7,
                )
                priority_delta += 8
            else:
                state.drive_vector["caution"] = max(
                    state.drive_vector.get("caution", 0.0),
                    0.7,
                )
                priority_delta -= 4
        return priority_delta

    def _apply_desires(
        self,
        desire_signals: tuple[str, ...],
        *,
        state: _IntentBuildState,
    ) -> int:
        priority_delta = 0
        for desire in desire_signals:
            cleaned = _clean_non_blank(desire, field_name="desire")
            normalized = cleaned.lower()
            state.drivers.append(f"desire:{cleaned[:48]}")
            state.evidence.append(f"desire:{cleaned}")
            if _contains_any(normalized, _SOCIAL_TERMS):
                state.drive_vector["social_calibration"] = max(
                    state.drive_vector.get("social_calibration", 0.0),
                    0.8,
                )
                state.data_needs.add(AgencyDataNeed.HUMAN_OPINION)
                state.outcomes.add(AgencyOutcomeKind.ASK_NIKITA)
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.TELEGRAM_MCP_READ,
                        capability="telegram_mcp_read",
                        target_id="social_context",
                        description="Сначала читать социальный контекст без ответа.",
                    ),
                )
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.TELEGRAM_MCP_SEND,
                        capability="telegram_mcp_send",
                        target_id="social_context",
                        description="Спросить человеческое мнение только после grant/judgement.",
                        side_effect=True,
                        permission_scope=SocialPermissionScope.REPLY_IF_ADDRESSED,
                    ),
                )
                state.safety_constraints.append(
                    "Social/outbound actions require scoped grant, judgement, rate and privacy gates."
                )
                priority_delta += 12
            if _contains_any(normalized, _BUDGET_TERMS):
                state.drive_vector["budget_care"] = max(
                    state.drive_vector.get("budget_care", 0.0),
                    0.8,
                )
                state.safety_constraints.append(
                    "low_arousal_budget_guard: budget/effort policy must gate autonomous loops."
                )
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.DRAFT_MESSAGE,
                        capability="agency_social_permission_request",
                        description="Сформулировать budget-aware черновик следующего шага.",
                    ),
                )
                priority_delta -= 8
        return priority_delta

    def _apply_learning_signals(
        self,
        learning_signals: tuple[LearningSignal, ...],
        *,
        state: _IntentBuildState,
    ) -> int:
        priority_delta = 0
        for signal in learning_signals:
            state.drivers.append(f"feedback:{signal.type}")
            state.evidence.append(f"learning:{signal.scope}:{signal.statement}")
            state.outcomes.add(AgencyOutcomeKind.MEMORY_CANDIDATE)
            state.data_needs.add(AgencyDataNeed.MEMORY)
            state.safety_constraints.append(
                "Feedback and memory/personality changes go through staging/spec gates."
            )
            _append_action(
                state.actions,
                AgencyAction(
                    kind=AgencyActionKind.STAGE_MEMORY,
                    capability="agency_stage_memory",
                    description=signal.statement,
                ),
            )
            if signal.type in {"correction", "boundary"}:
                state.outcomes.add(AgencyOutcomeKind.SPEC)
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.CREATE_SPEC,
                        capability="request_tier3_specs_for_nikita_approval",
                        description="Сделать feedback проверяемым контрактом.",
                    ),
                )
                priority_delta += int(signal.confidence * 10)
        return priority_delta

    def _apply_daemon_signals(
        self,
        daemon_signals: tuple[Signal, ...],
        *,
        state: _IntentBuildState,
    ) -> int:
        priority_delta = 0
        for signal in daemon_signals:
            state.drivers.append(f"daemon:{signal.signal_type}")
            state.evidence.append(f"daemon:{signal.source}:{signal.signal_type}")
            if signal.signal_type in {"topic_cluster_ready", "runtime_gap"}:
                state.outcomes.add(AgencyOutcomeKind.SPEC)
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.WEB_RESEARCH,
                        capability="web_search_sources",
                        description="Проверить внешний/текущий контекст read-only.",
                    ),
                )
                _append_action(
                    state.actions,
                    AgencyAction(
                        kind=AgencyActionKind.CREATE_SPEC,
                        capability="request_tier3_specs_for_nikita_approval",
                        description="Подготовить spec из daemon signal.",
                    ),
                )
                priority_delta += 10
            if signal.priority == "critical" or signal.requires_response:
                priority_delta += 8
        return priority_delta


def _append_action(actions: list[AgencyAction], action: AgencyAction) -> None:
    key = (action.kind, action.capability, action.target_id)
    if any((item.kind, item.capability, item.target_id) == key for item in actions):
        return
    actions.append(action)


def _contains_any(text: str, terms: frozenset[str]) -> bool:
    return any(term in text for term in terms)


def _finalize_state(state: _IntentBuildState) -> None:
    if not state.drivers:
        state.drivers.append("agency:manual_event")
    if not state.evidence:
        state.evidence.append("agency:event")
    if AgencyActionKind.WEB_RESEARCH not in {action.kind for action in state.actions}:
        _append_action(
            state.actions,
            AgencyAction(
                kind=AgencyActionKind.SEARCH_KB,
                capability="read_workspace",
                description="Сверить intent с накопленным knowledge/workspace контекстом.",
            ),
        )


def _intent_from_state(*, goal: str, state: _IntentBuildState) -> AgencyIntent:
    return AgencyIntent(
        kind=AgencyIntentKind.SELF_COMPLEXIFICATION,
        source="personality_driven_agency",
        goal=goal,
        why_complexification=(
            "Внутренние сигналы Жвуши требуют превратить разрыв в bounded "
            "intent, а не в prompt-only желание."
        ),
        why_personality_matters=_why_personality_matters(
            drive_vector=state.drive_vector,
            drivers=state.drivers,
        ),
        priority=_clamp_priority(state.priority),
        drive_vector=state.drive_vector,
        personality_drivers=tuple(state.drivers),
        safety_constraints=tuple(dict.fromkeys(state.safety_constraints)),
        data_needs=tuple(sorted(state.data_needs, key=lambda item: item.value)),
        candidate_actions=tuple(state.actions),
        expected_outcomes=tuple(sorted(state.outcomes, key=lambda item: item.value)),
        evidence=tuple(dict.fromkeys(state.evidence)),
    )


def _clean_non_blank(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-blank")
    return cleaned


def _clamp_priority(value: int) -> int:
    return max(0, min(100, value))


def _why_personality_matters(
    *,
    drive_vector: dict[str, float],
    drivers: list[str],
) -> str:
    top_drives = ", ".join(
        key
        for key, _value in sorted(
            drive_vector.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
    )
    return (
        "Intent выбран не из внешней команды alone: его двигают "
        f"{top_drives or 'self_complexification'}; signals={len(drivers)}."
    )
