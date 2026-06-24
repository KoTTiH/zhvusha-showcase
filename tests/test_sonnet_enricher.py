"""Tests for SonnetEnricher — async metadata extraction from user messages."""

from __future__ import annotations

import typing
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from src.llm.protocols import LLMResponse, LLMUsage
from src.memory.sonnet_enricher import (
    _ENRICHER_SYSTEM_PROMPT,
    EnrichmentResult,
    LearningSignal,
    SonnetEnricher,
    _strip_markdown,
)

_ROUTER_PATCH = "src.memory.pipelines.enrichment.get_router"


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


def _valid_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "importance": 0.7,
        "valence": "positive",
        "intent": "statement",
        "emotion": "happy",
        "confidence": 0.85,
        "is_feedback": False,
        "feedback_strength": 0.0,
        "reasoning": "Nikita shares a mildly positive update.",
        "learning_signal": None,
    }
    base.update(overrides)
    return base


def _valid_learning_signal(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": "rule",
        "statement": "не писать формально в personal mode",
        "scope": "tone",
        "confidence": 0.92,
        "apply_immediately": True,
        "original_claim": None,
    }
    base.update(overrides)
    return base


# --- Pydantic model validation ---


def test_enrichment_result_accepts_valid_payload() -> None:
    result = EnrichmentResult(**_valid_payload())  # type: ignore[arg-type]
    assert result.importance == 0.7
    assert result.valence == "positive"
    assert result.intent == "statement"
    assert result.emotion == "happy"
    assert result.confidence == 0.85
    assert result.is_feedback is False
    # New fields default correctly
    assert result.arousal == 0.5
    assert result.self_emotion == "curiosity"
    assert result.self_arousal == 0.5
    assert result.feedback_strength == 0.0
    assert "positive" in result.reasoning.lower()


def test_enrichment_result_accepts_explicit_new_fields() -> None:
    result = EnrichmentResult(
        **_valid_payload(  # type: ignore[arg-type]
            arousal=0.8,
            self_emotion="excitement",
            self_arousal=0.7,
        )
    )
    assert result.arousal == 0.8
    assert result.self_emotion == "excitement"
    assert result.self_arousal == 0.7


def test_enrichment_result_rejects_arousal_out_of_range() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(arousal=1.5))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(arousal=-0.1))  # type: ignore[arg-type]


def test_enrichment_result_rejects_unknown_self_emotion() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(self_emotion="rage"))  # type: ignore[arg-type]


def test_enrichment_result_rejects_self_arousal_out_of_range() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(self_arousal=2.0))  # type: ignore[arg-type]


def test_enricher_prompt_contains_arousal_criteria() -> None:
    assert "arousal" in _ENRICHER_SYSTEM_PROMPT
    assert "0.0-0.3" in _ENRICHER_SYSTEM_PROMPT
    assert "0.9-1.0" in _ENRICHER_SYSTEM_PROMPT


def test_enricher_prompt_contains_self_emotion_criteria() -> None:
    assert "self_emotion" in _ENRICHER_SYSTEM_PROMPT
    assert "self_arousal" in _ENRICHER_SYSTEM_PROMPT
    assert "curiosity" in _ENRICHER_SYSTEM_PROMPT
    assert "warmth" in _ENRICHER_SYSTEM_PROMPT


def test_enrichment_result_rejects_importance_above_one() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(importance=1.5))  # type: ignore[arg-type]


def test_enrichment_result_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(intent="gossip"))  # type: ignore[arg-type]


def test_enrichment_result_rejects_feedback_strength_out_of_range() -> None:
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(feedback_strength=2.0))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EnrichmentResult(**_valid_payload(feedback_strength=-1.5))  # type: ignore[arg-type]


# --- LearningSignal ---


def test_learning_signal_accepts_valid_rule_payload() -> None:
    signal = LearningSignal(**_valid_learning_signal())  # type: ignore[arg-type]
    assert signal.type == "rule"
    assert signal.scope == "tone"
    assert signal.confidence == 0.92
    assert signal.apply_immediately is True
    assert signal.original_claim is None
    assert "формально" in signal.statement


def test_learning_signal_accepts_correction_with_original_claim() -> None:
    signal = LearningSignal(
        **_valid_learning_signal(  # type: ignore[arg-type]
            type="correction",
            statement="на самом деле kwork — это единственный доход",
            scope="personal_facts",
            original_claim="Никита упоминал основную работу",
        )
    )
    assert signal.type == "correction"
    assert signal.original_claim == "Никита упоминал основную работу"


def test_learning_signal_rejects_correction_without_original_claim() -> None:
    with pytest.raises(ValidationError):
        LearningSignal(
            **_valid_learning_signal(  # type: ignore[arg-type]
                type="correction",
                original_claim=None,
            )
        )


def test_learning_signal_rejects_original_claim_for_non_correction() -> None:
    with pytest.raises(ValidationError):
        LearningSignal(
            **_valid_learning_signal(  # type: ignore[arg-type]
                type="rule",
                original_claim="что-то не то",
            )
        )


def test_learning_signal_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        LearningSignal(**_valid_learning_signal(type="gossip"))  # type: ignore[arg-type]


def test_learning_signal_rejects_unknown_scope() -> None:
    with pytest.raises(ValidationError):
        LearningSignal(**_valid_learning_signal(scope="random"))  # type: ignore[arg-type]


def test_learning_signal_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        LearningSignal(**_valid_learning_signal(confidence=1.5))  # type: ignore[arg-type]


def test_enrichment_result_learning_signal_defaults_to_none() -> None:
    # When learning_signal key is absent, default is None (not required field)
    payload = _valid_payload()
    del payload["learning_signal"]
    result = EnrichmentResult(**payload)  # type: ignore[arg-type]
    assert result.learning_signal is None


def test_enrichment_result_accepts_embedded_learning_signal() -> None:
    result = EnrichmentResult(
        **_valid_payload(learning_signal=_valid_learning_signal())  # type: ignore[arg-type]
    )
    assert result.learning_signal is not None
    assert result.learning_signal.type == "rule"
    assert result.learning_signal.scope == "tone"


# --- _strip_markdown ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ('  {"a": 1}  ', '{"a": 1}'),
        ('```json\n{"a":1}```', '{"a":1}'),
    ],
)
def test_strip_markdown_handles_json_bare_and_plain(raw: str, expected: str) -> None:
    assert _strip_markdown(raw) == expected


# --- SonnetEnricher.enrich() ---


async def test_enrich_calls_router_with_configured_tier_and_temperature_zero() -> None:
    import json

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=_llm_resp(json.dumps(_valid_payload()))
    )

    with patch(_ROUTER_PATCH, return_value=mock_router):
        enricher = SonnetEnricher(tier="worker")
        result = await enricher.enrich(message="привет")

    assert result is not None
    mock_router.generate.assert_awaited_once()
    request = mock_router.generate.call_args.args[0]
    assert request.tier == "worker"
    assert request.temperature == 0.0
    assert request.system == _ENRICHER_SYSTEM_PROMPT


async def test_enrich_respects_custom_tier() -> None:
    import json

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=_llm_resp(json.dumps(_valid_payload()))
    )

    with patch(_ROUTER_PATCH, return_value=mock_router):
        enricher = SonnetEnricher(tier="analyst")
        await enricher.enrich(message="test")

    request = mock_router.generate.call_args.args[0]
    assert request.tier == "analyst"


async def test_enrich_parses_valid_response_with_markdown_fences() -> None:
    import json

    fenced = f"```json\n{json.dumps(_valid_payload(valence='negative'))}\n```"
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(fenced))

    with patch(_ROUTER_PATCH, return_value=mock_router):
        enricher = SonnetEnricher()
        result = await enricher.enrich(
            message="да заебала формальными советами",
            recent_context="Никита: как дела?\nЖвуша: формальный ответ",
            prev_bot_response="формальный ответ",
        )

    assert result is not None
    assert result.valence == "negative"
    request = mock_router.generate.call_args.args[0]
    prompt = request.prompt
    assert "<CURRENT_MESSAGE>" in prompt
    assert "да заебала" in prompt
    assert "<RECENT_CONVERSATION>" in prompt
    assert "<PREVIOUS_BOT_RESPONSE>" in prompt
    assert "формальный ответ" in prompt


@pytest.mark.parametrize(
    "scenario",
    ["router_raises", "invalid_json", "schema_mismatch"],
)
async def test_enrich_returns_none_on_failures(scenario: str) -> None:
    mock_router = AsyncMock()
    if scenario == "router_raises":
        mock_router.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
    elif scenario == "invalid_json":
        mock_router.generate = AsyncMock(
            return_value=_llm_resp("not json at all {oops}")
        )
    else:  # schema_mismatch
        mock_router.generate = AsyncMock(return_value=_llm_resp('{"importance": 0.5}'))

    with patch(_ROUTER_PATCH, return_value=mock_router):
        enricher = SonnetEnricher()
        result = await enricher.enrich(message="test")

    assert result is None


def test_enricher_prompt_contains_all_literal_values() -> None:
    """Drift protection: every Literal value in EnrichmentResult AND nested
    LearningSignal must be mentioned in the system prompt, otherwise Sonnet
    won't know to emit it."""
    checks: list[tuple[str, object]] = []
    er_fields = EnrichmentResult.model_fields
    for field_name in ("valence", "intent", "emotion", "self_emotion"):
        checks.append(
            (f"EnrichmentResult.{field_name}", er_fields[field_name].annotation)
        )

    ls_fields = LearningSignal.model_fields
    for field_name in ("type", "scope"):
        checks.append(
            (f"LearningSignal.{field_name}", ls_fields[field_name].annotation)
        )

    for label, annotation in checks:
        args = typing.get_args(annotation)
        assert args, f"{label} should be a Literal type with args"
        for literal_value in args:
            assert isinstance(literal_value, str)
            assert literal_value in _ENRICHER_SYSTEM_PROMPT, (
                f"Literal value {literal_value!r} from {label} "
                f"is not mentioned in _ENRICHER_SYSTEM_PROMPT — "
                f"Sonnet will not know to emit it"
            )
