"""Tests for ``chat_self_coding.intent_classifier`` (Phase 40).

The classifier maps a user message + chat-mode context to one of eight
canonical intents (create / show / approve / reject / run / exit /
status / other). It uses a two-tier strategy:

1. Fast keyword/phrase match — covers explicit commands ("делай", "выход",
   "не надо", "оформи план") without an LLM round-trip.
2. LLM fallback on ``worker`` tier — for ambiguous text. Mirrors the
   ``daemon_approval.classify_intent`` shape (worker tier, temperature
   0.0, named ``caller``).

AGENTS.md exception for approval classification explicitly allows this
two-tier approach (cost/latency tradeoff).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import pytest
from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


def _mock_llm(reply: str = "other") -> AsyncMock:
    """AsyncMock LLMRouter that returns ``reply`` from ``generate``."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_llm_resp(reply))
    return llm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TestStructure:
    def test_intent_enum_has_eight_canonical_values(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Intent

        expected = {
            "create_spec",
            "show_spec",
            "approve",
            "reject",
            "run_spec",
            "merge",
            "exit",
            "status",
            "other",
        }
        assert {i.value for i in Intent} == expected

    def test_stage_enum_has_five_canonical_values(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage

        expected = {"idle", "drafting", "pending_approval", "running", "done"}
        assert {s.value for s in Stage} == expected

    def test_classification_is_frozen(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassification,
        )

        cls = IntentClassification(intent=Intent.APPROVE, slug="foo")
        with pytest.raises(FrozenInstanceError):
            cls.intent = Intent.REJECT  # type: ignore[misc]

    def test_context_is_frozen(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            Stage,
        )

        ctx = IntentClassifierContext(text="hi", stage=Stage.IDLE)
        with pytest.raises(FrozenInstanceError):
            ctx.text = "bye"  # type: ignore[misc]

    def test_context_recent_messages_is_immutable_tuple_default(self) -> None:
        """recent_messages must default to an immutable tuple, not a list."""
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            Stage,
        )

        ctx = IntentClassifierContext(text="x", stage=Stage.IDLE)
        assert isinstance(ctx.recent_messages, tuple)
        assert ctx.recent_messages == ()


# ---------------------------------------------------------------------------
# Keyword fast match (must NOT call LLM)
# ---------------------------------------------------------------------------


class TestKeywordFastMatch:
    async def test_implementation_trigger_recognized_when_pending_spec_exists(
        self,
    ) -> None:
        """Phase 40 spec-mandated test.

        Input: «делай» when a pending-approval spec exists.
        Expected: ``Intent.APPROVE`` carrying the active slug. Keyword path
        — LLM must not be consulted.
        """
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(
            text="делай",
            stage=Stage.PENDING_APPROVAL,
            active_spec_slug="bug-investigation-preset",
        )
        result = await classifier(ctx)
        assert result.intent == Intent.APPROVE
        assert result.slug == "bug-investigation-preset"
        llm.generate.assert_not_awaited()

    async def test_exit_recognized_from_synonyms(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        for text in ("выход", "хватит", "всё", "финиш", "exit"):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.EXIT, f"missed: {text!r}"
        llm.generate.assert_not_awaited()

    async def test_reject_recognized_from_phrases(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        for text in ("не надо", "отмена", "отклоняю", "не сейчас"):
            ctx = IntentClassifierContext(
                text=text,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="x",
            )
            result = await classifier(ctx)
            assert result.intent == Intent.REJECT, f"missed: {text!r}"
        llm.generate.assert_not_awaited()

    async def test_show_recognized_from_inquiry(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="покажи план", stage=Stage.PENDING_APPROVAL)
        result = await classifier(ctx)
        assert result.intent == Intent.SHOW_SPEC
        llm.generate.assert_not_awaited()

    async def test_status_recognized_from_phrase(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="как идёт", stage=Stage.RUNNING)
        result = await classifier(ctx)
        assert result.intent == Intent.STATUS
        llm.generate.assert_not_awaited()

    async def test_merge_recognized_only_when_done(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        done = await classifier(
            IntentClassifierContext(
                text="слей",
                stage=Stage.DONE,
                active_spec_slug="my-spec",
            )
        )
        pending = await classifier(
            IntentClassifierContext(
                text="слей",
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        assert done.intent == Intent.MERGE
        assert pending.intent == Intent.OTHER

    async def test_generic_action_words_do_not_create_spec_by_default(self) -> None:
        """Generic action words must not skip discussion and build a spec.

        The user can turn discussion into a spec with plan/spec wording.
        """
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in ("сделай", "исправь", "почини", "реализуй", "делай"):
            idle_ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            idle_result = await classifier(idle_ctx)
            assert idle_result.intent == Intent.OTHER, f"created too early: {text!r}"

        llm.generate.assert_awaited()

    async def test_explicit_plan_keywords_create_spec(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "оформи план",
            "собери план",
            "сформулируй план",
            "пересобери план",
            "новый план",
            "создай spec",
        ):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.CREATE_SPEC, f"missed: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_long_explicit_plan_request_creates_spec_without_readonly_detour(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "Отлично, сделай spec по нашему рассуждению",
            "Сделай план по тому, что мы обсудили выше",
        ):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.CREATE_SPEC, f"missed: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_retry_after_architect_failure_recreates_spec(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        ctx = IntentClassifierContext(
            text="Ещё раз попробуй",
            stage=Stage.IDLE,
            recent_messages=(
                "Никита: Новый план",
                "Жвуша: Не получилось составить план. Причина: parse failed",
            ),
        )
        result = await classifier(ctx)

        assert result.intent == Intent.CREATE_SPEC
        llm.generate.assert_not_awaited()

    async def test_explicit_plan_keywords_can_create_new_spec_after_pending_plan(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        ctx = IntentClassifierContext(
            text="пересобери план",
            stage=Stage.PENDING_APPROVAL,
            active_spec_slug="old-spec",
            recent_messages=(
                "Никита: план не учитывает новый guard",
                "Жвуша: Тогда spec нужно пересобрать.",
            ),
        )
        result = await classifier(ctx)

        assert result.intent == Intent.CREATE_SPEC
        assert result.slug == "old-spec"
        llm.generate.assert_not_awaited()

    async def test_idle_idea_messages_stay_discussion_until_explicit_plan(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("create_spec")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "хочу сделать правило для бытовых приветствий",
            "надо чтобы она обсуждала идеи",
            "нужно сделать поведение живее",
            "проблема в тоне ответа",
            "что если хранить сессию самокодинга?",
        ):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.OTHER, f"created too early: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_fix_request_keywords_do_not_create_spec_mid_cycle(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        ctx = IntentClassifierContext(
            text="исправь это поведение",
            stage=Stage.RUNNING,
            active_spec_slug="some-spec",
        )
        result = await classifier(ctx)
        assert result.intent != Intent.CREATE_SPEC

    async def test_long_messages_with_command_words_stay_discussion(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("exit")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "если я напишу слово выход будет триггер",
            "в сообщении есть слово делай но это пример",
            "не надо тут воспринимать как команду",
        ):
            result = await classifier(
                IntentClassifierContext(text=text, stage=Stage.IDLE)
            )
            assert result.intent == Intent.OTHER, f"false command: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_pasted_context_block_waits_for_explicit_plan_in_idle(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(
            text=(
                "Контекст:\n"
                '"Никита: Доброе утро"\n'
                '"Жвуша: Доброе утро, Никита. Я тут, проснулась рядом..."\n\n'
                "Нужно сдвинуть правило приветствий ближе к прежней личности."
            ),
            stage=Stage.IDLE,
        )
        result = await classifier(ctx)
        assert result.intent == Intent.OTHER
        llm.generate.assert_not_awaited()

    async def test_discussion_phrase_does_not_fast_create_spec_in_idle(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "хочу обсудить как исправить приветствие",
            "что думаешь про эту проблему",
            "как лучше это сделать?",
        ):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.OTHER, f"missed discussion: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_finalize_discussion_phrases_create_spec_in_idle(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in (
            "оформи план",
            "сформулируй план",
            "пересобери план",
            "создай spec",
        ):
            ctx = IntentClassifierContext(text=text, stage=Stage.IDLE)
            result = await classifier(ctx)
            assert result.intent == Intent.CREATE_SPEC, f"missed finalize: {text!r}"

        llm.generate.assert_not_awaited()

    async def test_pending_plan_defaults_to_discussion_until_run_trigger(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("approve")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in ("давай", "ок", "да", "ага", "ну ладно"):
            ctx = IntentClassifierContext(
                text=text,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
            result = await classifier(ctx)
            assert result.intent == Intent.OTHER, f"ran too early: {text!r}"

    async def test_run_triggers_only_approve_after_pending_plan(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)

        for text in ("делай", "реализуй", "запускай", "пиши код"):
            result = await classifier(
                IntentClassifierContext(
                    text=text,
                    stage=Stage.PENDING_APPROVAL,
                    active_spec_slug="my-spec",
                )
            )
            assert result.intent == Intent.APPROVE, f"missed run trigger: {text!r}"

        idle = await classifier(IntentClassifierContext(text="делай", stage=Stage.IDLE))
        assert idle.intent == Intent.OTHER

    async def test_tier3_approval_skips_keyword_path_and_uses_llm(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("approve")
        classifier = LLMIntentClassifier(llm_router=llm)

        result = await classifier(
            IntentClassifierContext(
                text="делай",
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="tier3-spec",
                requires_ai_approval=True,
            )
        )

        assert result.intent == Intent.APPROVE
        assert result.reasoning == "llm classification"
        llm.generate.assert_awaited_once()

    async def test_keyword_results_have_high_confidence(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm()
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="одобряю", stage=Stage.PENDING_APPROVAL)
        result = await classifier(ctx)
        assert result.confidence >= 0.9


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


class TestLLMFallback:
    async def test_explicit_natural_run_request_can_fall_through_to_llm(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("approve")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(
            text="можешь начинать реализацию",
            stage=Stage.PENDING_APPROVAL,
            active_spec_slug="some-spec",
        )
        result = await classifier(ctx)
        assert result.intent == Intent.APPROVE
        assert result.slug == "some-spec"
        llm.generate.assert_awaited_once()

    async def test_llm_unrecognized_token_returns_other(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("garbage-not-an-intent")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="что-то странное", stage=Stage.IDLE)
        result = await classifier(ctx)
        assert result.intent == Intent.OTHER

    async def test_uses_worker_tier_with_zero_temperature(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("status")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="мм", stage=Stage.RUNNING)
        await classifier(ctx)
        request = llm.generate.call_args.args[0]
        assert request.tier == "worker"
        assert request.temperature == 0.0
        assert request.caller == "chat_self_coding_intent"

    async def test_llm_prompt_includes_stage_and_slug_context(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("show_spec")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(
            text="ну",
            stage=Stage.RUNNING,
            active_spec_slug="my-spec",
        )
        await classifier(ctx)
        request = llm.generate.call_args.args[0]
        combined = (request.system or "") + " " + request.prompt
        assert "running" in combined.lower()
        assert "my-spec" in combined

    async def test_recent_messages_passed_to_llm(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("other")
        classifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(
            text="посмотрю",  # not a keyword — forces LLM fallback
            stage=Stage.DONE,
            active_spec_slug="x",
            recent_messages=("Жвуша: Что делаем?", "Никита: подумаю"),
        )
        await classifier(ctx)
        request = llm.generate.call_args.args[0]
        assert "подумаю" in request.prompt

    async def test_llm_response_is_case_insensitive(self) -> None:
        """LLM may return 'Approve' or 'APPROVE' — both should map."""
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        for reply in ("Approve", "APPROVE", "approve"):
            llm = _mock_llm(reply)
            classifier = LLMIntentClassifier(llm_router=llm)
            # This phrase is intentionally outside the keyword shortcut so the
            # LLM is consulted, but it is still an explicit implementation request.
            ctx = IntentClassifierContext(
                text="можешь начинать реализацию",
                stage=Stage.PENDING_APPROVAL,
            )
            result = await classifier(ctx)
            assert result.intent == Intent.APPROVE, f"failed reply={reply!r}"


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


class TestProtocol:
    async def test_classifier_satisfies_protocol(self) -> None:
        """LLMIntentClassifier must be usable as IntentClassifier (structural)."""
        from src.skills.chat_self_coding.intent_classifier import (
            IntentClassifier,
            IntentClassifierContext,
            LLMIntentClassifier,
            Stage,
        )

        llm = _mock_llm("exit")
        classifier: IntentClassifier = LLMIntentClassifier(llm_router=llm)
        ctx = IntentClassifierContext(text="выход", stage=Stage.IDLE)
        result = await classifier(ctx)
        assert result is not None
