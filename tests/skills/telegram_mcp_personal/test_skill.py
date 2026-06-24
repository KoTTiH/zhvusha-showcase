"""Telegram MCP personal skill contract tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.agent_runtime.models import AgentJobStatus
from src.agent_runtime.profiles import TELEGRAM_MCP_PERSONAL_ACTIONS
from src.skills.base import AgentContext
from src.skills.invocation import (
    ApprovalVerdict,
    InMemorySkillApprovalStore,
    SkillInvocationService,
)
from src.skills.telegram_mcp_personal.skill import TelegramMCPActionIntent


class _Runtime:
    def __init__(self, *, result: object | None = None) -> None:
        self.created: list[dict[str, object]] = []
        self._result = result

    async def create_job(self, **kwargs: object) -> object:
        self.created.append(kwargs)
        return SimpleNamespace(id="job-telegram")

    async def start(self, job_id: str) -> object:
        assert job_id == "job-telegram"
        return SimpleNamespace(
            id=job_id,
            status=AgentJobStatus.DONE,
            result=self._result
            or SimpleNamespace(summary="Telegram MCP action completed."),
            error="",
        )


class _IntentClassifier:
    def __init__(self, *intents: TelegramMCPActionIntent) -> None:
        self._intents = list(intents)
        self.calls: list[str] = []

    async def classify(self, message: str) -> TelegramMCPActionIntent:
        self.calls.append(message)
        if self._intents:
            return self._intents.pop(0)
        return TelegramMCPActionIntent(action="none", confidence=0.0)


class _HangingIntentClassifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def classify(self, message: str) -> TelegramMCPActionIntent:
        self.calls.append(message)
        await asyncio.sleep(60.0)
        return TelegramMCPActionIntent(action="none", confidence=0.0)


def _ctx(
    *,
    approved: bool = False,
    metadata: dict[str, object] | None = None,
) -> AgentContext:
    context_metadata = dict(metadata or {})
    if approved:
        context_metadata.update(
            {
                "skill_approval_granted": True,
                "skill_approval_id": "skill-approval-test",
            }
        )
    return AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=5,
        metadata=context_metadata,
    )


def _service(*verdicts: ApprovalVerdict) -> SkillInvocationService:
    queued_verdicts = list(verdicts or ("yes",))

    async def classify_approval(text: str) -> ApprovalVerdict:
        del text
        if queued_verdicts:
            return queued_verdicts.pop(0)
        return "yes"

    return SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify_approval,
        is_skill_allowed=lambda _name, _mode: True,
    )


def test_manifest_matches_class() -> None:
    from src.skills.manifest import (
        load_manifest_for_skill_class,
        validate_manifest_matches_class,
    )
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    manifest = load_manifest_for_skill_class(TelegramMCPPersonalSkill)
    validate_manifest_matches_class(manifest, TelegramMCPPersonalSkill)


@pytest.mark.asyncio
async def test_send_command_requires_approval_and_starts_runtime_job_after_approval() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
    )

    assert await skill.can_handle("/telegram_send @nikita | привет", _ctx()) == 0.95
    plan = await skill.prepare("/telegram_send @nikita | привет", _ctx())
    assert plan.human_summary == (
        "Отправить личное Telegram сообщение в @nikita: «привет»"
    )
    simulation = await skill.dry_run(plan)
    assert simulation.dependencies_available is True
    assert simulation.would_succeed is True

    result = await skill.execute("/telegram_send @nikita | привет", _ctx(approved=True))

    assert result.success is True
    assert result.metadata["agent_job_id"] == "job-telegram"
    created = runtime.created[0]
    assert created["profile"] == TELEGRAM_MCP_PERSONAL_ACTIONS
    pack = created["context_pack"]
    assert pack.metadata["agent_tool_approval_id"] == "skill-approval-test"
    assert pack.metadata["agent_tool_approval_capabilities"] == "telegram_mcp_send"


@pytest.mark.asyncio
async def test_read_command_uses_readonly_profile() -> None:
    from src.agent_runtime.profiles import TELEGRAM_MCP_PERSONAL_READONLY
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        readonly_profile=TELEGRAM_MCP_PERSONAL_READONLY,
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
    )

    assert await skill.can_handle("/telegram_read @nikita 3", _ctx()) == 0.95
    result = await skill.execute("/telegram_read @nikita 3", _ctx())

    assert result.success is True
    created = runtime.created[0]
    assert created["profile"] == TELEGRAM_MCP_PERSONAL_READONLY


@pytest.mark.asyncio
async def test_natural_language_send_intent_routes_above_chat_fallback_score() -> None:
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="@KoTTiH",
            message="привет",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=5,
        metadata={
            "recent_messages": ["Жвуша: пока не могу, личный Telegram MCP не подключён"]
        },
    )
    message = "Через личный Telegram отправь Никите: привет"

    score = await skill.can_handle(message, context)

    assert score > 0.3
    assert classifier.calls[0].endswith(message)
    assert "Недавний диалог" in classifier.calls[0]
    plan = await skill.prepare(message, context)
    assert plan.human_summary == (
        "Отправить личное Telegram сообщение в @KoTTiH: «привет»"
    )

    result = await skill.execute(message, _ctx(approved=True))

    assert result.success is True
    created = runtime.created[0]
    pack = created["context_pack"]
    assert '"action": "send_message"' in pack.user_request
    assert '"chat_id": "@KoTTiH"' in pack.user_request
    assert '"message": "привет"' in pack.user_request


@pytest.mark.asyncio
async def test_natural_language_classifier_timeout_falls_back_to_chat() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    classifier = _HangingIntentClassifier()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
        intent_timeout_seconds=0.01,
    )

    score = await asyncio.wait_for(
        skill.can_handle("как дела у тебя?", _ctx()),
        timeout=0.5,
    )

    assert score == 0.0
    assert classifier.calls
    assert runtime.created == []


@pytest.mark.asyncio
async def test_codex_author_marker_does_not_skip_telegram_mcp_classifier() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.95,
            chat_id="@KoTTiH",
            message="не должно использоваться",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )

    score = await skill.can_handle(
        "скинь Тоше ок",
        _ctx(metadata={"source_actor": "codex"}),
    )

    assert score == 0.91
    assert classifier.calls == ["скинь Тоше ок"]
    assert runtime.created == []


@pytest.mark.asyncio
async def test_fast_chat_preselection_does_not_run_telegram_mcp_classifier() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.95,
            chat_id="@KoTTiH",
            message="не должно использоваться",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )

    score = await skill.can_handle(
        "как дела у тебя?",
        _ctx(metadata={"prefer_chat_response_only": True}),
    )

    assert score == 0.0
    assert classifier.calls == []
    assert runtime.created == []


@pytest.mark.asyncio
async def test_natural_language_send_intent_asks_short_clarification_then_approval() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.88,
            chat_id="@KoTTiH",
            message="",
            missing_fields=("message",),
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    clarification = await service.dispatch(
        "Напиши Никите через личный Telegram",
        _ctx(),
        [skill],
    )

    assert clarification.handled is True
    assert clarification.result is not None
    assert clarification.result.response == "Что написать в @KoTTiH?"
    assert clarification.result.metadata["requires_user_input"] is True
    assert runtime.created == []

    approval = await service.dispatch("Напиши: привет", _ctx(), [skill])

    assert approval.handled is True
    assert approval.result is not None
    assert "Нужно решение" in approval.result.response
    assert approval.result.metadata["requires_zhvusha_response"] is True
    assert runtime.created == []

    executed = await service.dispatch("да", _ctx(), [skill])

    assert executed.handled is True
    assert executed.result is not None
    assert executed.result.success is True
    pack = runtime.created[0]["context_pack"]
    assert '"message": "Напиши: привет"' in pack.user_request


@pytest.mark.asyncio
async def test_pending_clarification_meta_question_is_not_used_as_message_text() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="Тоше",
            message="",
            missing_fields=("message",),
        ),
        TelegramMCPActionIntent(action="none", confidence=0.92),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.91,
            chat_id="@Anroxa2748",
            message="привет",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    clarification = await service.dispatch("Напишешь тохе?", _ctx(), [skill])

    assert clarification.result is not None
    assert clarification.result.response == (
        "Не хватает @username/id для Тоше и текста сообщения."
    )
    assert clarification.result.metadata["requires_zhvusha_response"] is True
    assert "в тохе" not in clarification.result.response.lower()

    meta_reply = await service.dispatch('Почему "в"?', _ctx(), [skill])

    assert meta_reply.result is not None
    assert meta_reply.result.response == (
        "Не хватает @username/id для Тоше и текста сообщения."
    )
    assert "Нужно решение" not in meta_reply.result.response
    assert 'Почему "в"?' not in meta_reply.result.response
    assert runtime.created == []

    approval = await service.dispatch("Напиши: привет", _ctx(), [skill])

    assert approval.result is not None
    assert "Нужно решение" in approval.result.response
    assert "привет" in approval.result.response
    assert 'Почему "в"?' not in approval.result.response


@pytest.mark.asyncio
async def test_pending_human_recipient_replies_do_not_reset_to_generic_question() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="Тоше",
            message="",
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="Тоше",
            message="",
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="Тоше",
            message="",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    first = await service.dispatch("Напишешь Тоше?", _ctx(), [skill])
    second = await service.dispatch("Тоше", _ctx(), [skill])
    third = await service.dispatch("Тоше написать", _ctx(), [skill])

    assert first.result is not None
    assert second.result is not None
    assert third.result is not None
    expected = "Не хватает @username/id для Тоше и текста сообщения."
    assert first.result.response == expected
    assert second.result.response == expected
    assert third.result.response == expected
    assert runtime.created == []


@pytest.mark.asyncio
async def test_pending_recipient_correction_drops_previous_creator_chat_id() -> None:
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="@KoTTiH",
            message="",
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.91,
            chat_id="Тоше",
            message="",
        ),
        TelegramMCPActionIntent(action="none", confidence=0.9),
        TelegramMCPActionIntent(action="none", confidence=0.9),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    first = await service.dispatch("Теперь напишешь?", _ctx(), [skill])
    correction = await service.dispatch("Не мне, а Тоше", _ctx(), [skill])
    username_meta = await service.dispatch("Это мой юсернейм", _ctx(), [skill])
    frustration = await service.dispatch("Ты не вдупляешь?", _ctx(), [skill])

    assert first.result is not None
    assert first.result.response == "Что написать в @KoTTiH?"
    assert correction.result is not None
    assert username_meta.result is not None
    assert frustration.result is not None
    expected = "Не хватает @username/id для Тоше и текста сообщения."
    assert correction.result.response == expected
    assert username_meta.result.response == expected
    assert frustration.result.response == expected
    assert "@KoTTiH" not in correction.result.response
    assert "@KoTTiH" not in username_meta.result.response
    assert "@KoTTiH" not in frustration.result.response
    assert runtime.created == []


@pytest.mark.asyncio
async def test_pronoun_send_followup_uses_recent_recipient_hint_and_draft_not_creator() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            recipient_hint="Тоше",
            message="Тош, не расстраивайся. Дота жива, пока ты живой.",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()
    recent_messages = "\n".join(
        (
            "Собеседник: Не мне, а Тоше",
            "Собеседник: Это мой юсернейм",
            (
                "Жвуша: Да. Теперь вдуплила: не тебе, а Тоше, и это твой "
                "юзернейм как адресат.\n\n"
                "Написала бы так:\n\n"
                "Тош, не расстраивайся. Дота жива, пока ты живой."
            ),
        )
    )

    outcome = await service.dispatch(
        "Пиши ему",
        _ctx(metadata={"recent_messages": recent_messages}),
        [skill],
    )

    assert outcome.result is not None
    assert outcome.result.response == "Не хватает @username/id для Тоше."
    assert "@KoTTiH" not in outcome.result.response
    assert runtime.created == []
    assert "Собеседник: Не мне, а Тоше" in classifier.calls[-1]
    assert "предложила черновик сообщения" in classifier.calls[-1]


@pytest.mark.asyncio
async def test_pronoun_send_followup_prefers_dialogue_state_over_stale_history() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            recipient_hint="Тоше",
            message="Тош, не расстраивайся. Дота жива, пока ты живой.",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    outcome = await service.dispatch(
        "Пиши ему",
        _ctx(
            metadata={
                "dialogue_state": {
                    "pending_action": "telegram_send",
                    "selected_skill": "telegram_mcp_personal",
                    "recipient_hint": "Тоше",
                    "executable_chat_id": "",
                    "draft_message": (
                        "Тош, не расстраивайся. Дота жива, пока ты живой."
                    ),
                    "missing_fields": ["chat_id"],
                },
                "dialogue_context": (
                    "pending_action: telegram_send\n"
                    "recipient_hint: Тоше\n"
                    "executable_chat_id: missing\n"
                    "draft_message: Тош, не расстраивайся. Дота жива, пока ты живой.\n"
                    "missing_fields: chat_id"
                ),
                "recent_decision_messages": (
                    "Собеседник: Не мне, а Тоше\nЖвуша: Не хватает @username/id."
                ),
                "recent_messages": "Жвуша: Что написать в @KoTTiH?",
            }
        ),
        [skill],
    )

    assert outcome.result is not None
    assert outcome.result.response == "Не хватает @username/id для Тоше."
    assert "@KoTTiH" not in outcome.result.response
    assert runtime.created == []
    prompt = classifier.calls[-1]
    assert "<DIALOGUE_STATE>" in prompt
    assert "recipient_hint: Тоше" in prompt
    assert "<RECENT_DECISION_CONTEXT>" in prompt
    assert "Жвуша: Что написать в @KoTTiH?" not in prompt


@pytest.mark.asyncio
async def test_alias_lookup_proposes_known_username_without_execution(
    tmp_path: Path,
) -> None:
    from src.dialogue.people import (
        FilePeopleAliasStore,
        extract_people_alias_candidates,
    )
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    people_alias_store = FilePeopleAliasStore(tmp_path)
    people_alias_store.append(
        chat_id=12345,
        candidate=extract_people_alias_candidates(
            "@Anroxa2748 это Тоша",
            source_message_id="tg:10",
        )[0],
    )
    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            recipient_hint="Тоше",
            message="Тош, не расстраивайся.",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
        people_alias_store=people_alias_store,
    )
    service = _service()

    outcome = await service.dispatch("Пиши Тоше", _ctx(), [skill])

    assert outcome.result is not None
    assert "Тоше" in outcome.result.response
    assert "@Anroxa2748" in outcome.result.response
    assert runtime.created == []

    metadata = outcome.result.metadata
    assert metadata["missing_fields"] == ["chat_id"]
    assert metadata["suggested_telegram_recipient"] == "@Anroxa2748"
    assert metadata["people_alias_lookup"]["can_execute"] is False
    assert metadata["people_alias_lookup"]["missing_fields"] == ["chat_id"]
    patch = metadata["dialogue_state_patch"]
    assert patch["recipient_hint"] == "Тоше"
    assert patch["clear_executable_chat_id"] is True
    assert "executable_chat_id" not in patch


@pytest.mark.asyncio
async def test_alias_lookup_confirmation_promotes_suggestion_to_approval_plan(
    tmp_path: Path,
) -> None:
    from src.dialogue.people import (
        FilePeopleAliasStore,
        extract_people_alias_candidates,
    )
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    people_alias_store = FilePeopleAliasStore(tmp_path)
    people_alias_store.append(
        chat_id=12345,
        candidate=extract_people_alias_candidates(
            "@Anroxa2748 это Тоша",
            source_message_id="tg:10",
        )[0],
    )
    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            recipient_hint="Тоше",
            message="Тош, не расстраивайся.",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
        people_alias_store=people_alias_store,
    )
    service = _service("yes")

    suggestion = await service.dispatch("Пиши Тоше", _ctx(), [skill])
    assert suggestion.result is not None
    assert suggestion.result.metadata["requires_recipient_confirmation"] is True
    assert suggestion.result.metadata["suggested_telegram_recipient"] == "@Anroxa2748"

    approval = await service.dispatch("да, это он", _ctx(), [skill])

    assert approval.result is not None
    assert "Отправить личное Telegram сообщение в @Anroxa2748" in (
        approval.result.response
    )
    assert "Тош, не расстраивайся." in approval.result.response
    assert runtime.created == []

    executed = await service.dispatch("да", _ctx(), [skill])

    assert executed.result is not None
    assert executed.result.response == "отправила."
    created = runtime.created[0]
    assert created["profile"] == TELEGRAM_MCP_PERSONAL_ACTIONS
    pack = created["context_pack"]
    assert "@Anroxa2748" in pack.user_request
    assert "Тош, не расстраивайся." in pack.user_request


@pytest.mark.asyncio
async def test_alias_lookup_rejection_suppresses_same_suggestion_then_accepts_correction(
    tmp_path: Path,
) -> None:
    from src.dialogue.people import (
        FilePeopleAliasStore,
        extract_people_alias_candidates,
    )
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    people_alias_store = FilePeopleAliasStore(tmp_path)
    people_alias_store.append(
        chat_id=12345,
        candidate=extract_people_alias_candidates(
            "@Anroxa2748 это Тоша",
            source_message_id="tg:10",
        )[0],
    )
    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            recipient_hint="Тоше",
            message="Тош, не расстраивайся.",
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
        people_alias_store=people_alias_store,
    )
    service = _service("yes")

    suggestion = await service.dispatch("Пиши Тоше", _ctx(), [skill])
    rejection = await service.dispatch("нет, не он", _ctx(), [skill])
    correction = await service.dispatch("@OtherUser", _ctx(), [skill])

    assert suggestion.result is not None
    assert rejection.result is not None
    assert correction.result is not None
    assert rejection.result.response == (
        "Ок, не @Anroxa2748. Пришли @username/id для Тоше."
    )
    assert rejection.result.metadata["people_alias_lookup"]["status"] == "rejected"
    assert rejection.result.metadata["suggested_telegram_recipient"] == ""
    assert "source_text" not in rejection.result.metadata["people_alias_lookup_status"]
    assert "Отправить личное Telegram сообщение в @OtherUser" in (
        correction.result.response
    )
    assert runtime.created == []


@pytest.mark.asyncio
async def test_pending_creative_reply_uses_classifier_draft_not_literal_instruction() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.88,
            chat_id="@KoTTiH",
            message="",
            missing_fields=("message",),
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.91,
            chat_id="@KoTTiH",
            message="я сама выбрала: пусть у тебя сегодня будет чуть больше воздуха.",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service()

    await service.dispatch("Можешь написать мне со своего акка?", _ctx(), [skill])
    approval = await service.dispatch("придумай сама, от себя", _ctx(), [skill])

    assert approval.handled is True
    assert approval.result is not None
    assert "я сама выбрала" in approval.result.response
    assert "придумай сама" not in approval.result.response
    assert runtime.created == []

    executed = await service.dispatch("да", _ctx(), [skill])

    assert executed.result is not None
    assert executed.result.response == "отправила."
    assert len(classifier.calls) == 2
    assert "придумай сама, от себя" in classifier.calls[1]
    pack = runtime.created[0]["context_pack"]
    assert "я сама выбрала" in pack.user_request
    assert "придумай сама" not in pack.user_request


@pytest.mark.asyncio
async def test_followup_after_rejected_send_keeps_previous_recipient_context() -> None:
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="@Anroxa2748",
            message="",
            missing_fields=("message",),
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.91,
            chat_id="@Anroxa2748",
            message="черкаш",
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            chat_id="@Anroxa2748",
            message="я тут сама придумала тебе написать. как ты там, живой?",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service("no", "yes")

    clarification = await service.dispatch(
        "@Anroxa2748 это Тоша, черкани ему в личку",
        _ctx(),
        [skill],
    )
    assert clarification.result is not None
    assert clarification.result.response == "Что написать в @Anroxa2748?"

    rejected_approval = await service.dispatch("черкаш", _ctx(), [skill])
    assert rejected_approval.result is not None
    assert "Отправить личное Telegram сообщение в @Anroxa2748" in (
        rejected_approval.result.response
    )

    rejected = await service.dispatch("нет", _ctx(), [skill])
    assert rejected.result is not None
    assert "Не выполняю" in rejected.result.response
    assert runtime.created == []

    followup_approval = await service.dispatch(
        "напиши еще что-нибудь но от себя придумай",
        _ctx(),
        [skill],
    )

    assert followup_approval.result is not None
    assert "Отправить личное Telegram сообщение в @Anroxa2748" in (
        followup_approval.result.response
    )
    assert "@KoTTiH" not in followup_approval.result.response
    assert "previous_telegram_chat_id: @Anroxa2748" in classifier.calls[-1]

    executed = await service.dispatch("да", _ctx(), [skill])

    assert executed.result is not None
    assert executed.result.response == "отправила."
    pack = runtime.created[0]["context_pack"]
    assert '"chat_id": "@Anroxa2748"' in pack.user_request
    assert "сама придумала" in pack.user_request


@pytest.mark.asyncio
async def test_pending_approval_text_feedback_rewrites_draft_instead_of_yes_no_loop() -> (
    None
):
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.92,
            chat_id="@Anroxa2748",
            message=(
                "Тоха, не расстраивайся. Дота, конечно, при смерти, "
                "но мы ещё живы — значит, катка продолжается."
            ),
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.94,
            chat_id="@Anroxa2748",
            message="Тоха, не скисай. Дота переживала и худшие времена.",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service("ambiguous", "yes")

    approval = await service.dispatch(
        "Напиши чета @Anroxa2748 чтобы не расстраивался о доте",
        _ctx(),
        [skill],
    )
    assert approval.result is not None
    assert "катка продолжается" in approval.result.response

    revision = await service.dispatch("Плохой текст", _ctx(), [skill])

    assert revision.result is not None
    assert "Ответь явно" not in revision.result.response
    assert "Тоха, не скисай" in revision.result.response
    assert "катка продолжается" not in revision.result.response
    assert "Фидбек Никиты" in classifier.calls[-1]
    assert runtime.created == []

    executed = await service.dispatch("да", _ctx(), [skill])

    assert executed.result is not None
    assert executed.result.response == "отправила."
    pack = runtime.created[0]["context_pack"]
    assert '"chat_id": "@Anroxa2748"' in pack.user_request
    assert "Тоха, не скисай" in pack.user_request
    assert "катка продолжается" not in pack.user_request


@pytest.mark.asyncio
async def test_named_recipient_followup_keeps_recent_real_chat_id_context() -> None:
    from src.skills.telegram_mcp_personal.skill import (
        TelegramMCPActionIntent,
        TelegramMCPPersonalSkill,
    )

    runtime = _Runtime()
    classifier = _IntentClassifier(
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.9,
            chat_id="@Anroxa2748",
            message="первый текст",
        ),
        TelegramMCPActionIntent(
            action="send_message",
            confidence=0.91,
            chat_id="@Anroxa2748",
            message="Тоха, не расстраивайся из-за доты.",
        ),
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        intent_classifier=classifier,
    )
    service = _service("yes", "yes")

    await service.dispatch(
        "@Anroxa2748 это Тоша, напиши ему первый текст", _ctx(), [skill]
    )
    await service.dispatch("да", _ctx(), [skill])
    second = await service.dispatch(
        "Напиши чета тохе чтобы не расстраивался о доте",
        _ctx(),
        [skill],
    )

    assert second.result is not None
    assert "Отправить личное Telegram сообщение в @Anroxa2748" in second.result.response
    assert "previous_telegram_chat_id: @Anroxa2748" in classifier.calls[-1]
    assert "Человеческое имя без @username" in classifier.calls[-1]
    assert "Тоше" not in second.result.response


@pytest.mark.asyncio
async def test_send_success_response_is_human_not_raw_tool_payload() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime(
        result=SimpleNamespace(
            summary="Telegram MCP action completed.",
            processed_context='{"result": "Message sent successfully."}',
            markdown_report='{"result": "Message sent successfully."}',
        )
    )
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
    )

    result = await skill.execute("/telegram_send @nikita | привет", _ctx(approved=True))

    assert result.success is True
    assert result.response == "отправила."


@pytest.mark.asyncio
async def test_successful_social_send_records_grant_usage_after_runtime_done(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from src.agency.models import (
        SocialPermissionGrant,
        SocialPermissionScope,
        SocialTargetType,
    )
    from src.agency.store import FileSocialPermissionStore
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        window_seconds=3600,
    )
    store.add(grant)
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=_Runtime(),  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        social_send_recorder=store,
    )

    result = await skill.execute(
        "/telegram_send @devchat | runtime контекст",
        _ctx(
            approved=True,
            metadata={
                "social_send_gate_result": {
                    "allowed": True,
                    "grant_id": grant.id,
                    "target_id": "@devchat",
                }
            },
        ),
    )

    assert result.success is True
    assert result.metadata["social_send_recorded"] is True
    assert result.metadata["social_permission_grant_id"] == grant.id
    assert (
        store.count_sent_in_window(
            grant_id=grant.id,
            now=datetime.now(tz=UTC),
            window_seconds=grant.window_seconds,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_manual_send_without_social_gate_metadata_does_not_record_grant_usage(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from src.agency.models import (
        SocialPermissionGrant,
        SocialPermissionScope,
        SocialTargetType,
    )
    from src.agency.store import FileSocialPermissionStore
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        window_seconds=3600,
    )
    store.add(grant)
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=_Runtime(),  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        social_send_recorder=store,
    )

    result = await skill.execute(
        "/telegram_send @devchat | ручное сообщение",
        _ctx(approved=True),
    )

    assert result.success is True
    assert "social_send_recorded" not in result.metadata
    assert (
        store.count_sent_in_window(
            grant_id=grant.id,
            now=datetime.now(tz=UTC),
            window_seconds=grant.window_seconds,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_social_send_gate_block_stops_before_runtime_job() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
    )

    result = await skill.execute(
        "/telegram_send @devchat | нельзя",
        _ctx(
            approved=True,
            metadata={
                "social_send_gate_result": {
                    "allowed": False,
                    "reason": "missing_active_grant",
                    "target_id": "@devchat",
                }
            },
        ),
    )

    assert result.success is False
    assert "missing_active_grant" in result.response
    assert runtime.created == []


@pytest.mark.asyncio
async def test_social_send_gate_target_mismatch_stops_before_runtime_job() -> None:
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
    )

    result = await skill.execute(
        "/telegram_send @other | нельзя",
        _ctx(
            approved=True,
            metadata={
                "social_send_gate_result": {
                    "allowed": True,
                    "reason": "allowed_by_grant_and_judgement",
                    "target_id": "@devchat",
                    "grant_id": "grant-devchat",
                }
            },
        ),
    )

    assert result.success is False
    assert "target_mismatch" in result.response
    assert runtime.created == []


@pytest.mark.asyncio
async def test_autonomous_social_send_candidate_runs_live_gate_before_runtime_job(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from src.agency.models import (
        SocialPermissionGrant,
        SocialPermissionScope,
        SocialTargetType,
    )
    from src.agency.social_gate import SocialSendGate
    from src.agency.store import FileSocialPermissionStore
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        allowed_topics=("runtime",),
        max_messages_per_window=2,
        window_seconds=3600,
    )
    store.add(grant)
    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        social_send_gate=SocialSendGate(store=store),
        social_send_recorder=store,
    )

    result = await skill.execute(
        "/telegram_send @devchat | Добавлю контекст по runtime.",
        _ctx(
            approved=True,
            metadata={
                "social_send_candidate": True,
                "social_send_topic": "runtime",
                "social_send_addressed_to_zhvusha": True,
                "social_send_has_value_to_add": True,
            },
        ),
    )

    assert result.success is True
    assert len(runtime.created) == 1
    assert result.metadata["social_send_recorded"] is True
    assert result.metadata["social_permission_grant_id"] == grant.id
    assert (
        store.count_sent_in_window(
            grant_id=grant.id,
            now=datetime.now(tz=UTC),
            window_seconds=grant.window_seconds,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_autonomous_social_send_candidate_blocked_by_live_gate_before_runtime_job(
    tmp_path: Path,
) -> None:
    from src.agency.models import (
        SocialPermissionGrant,
        SocialPermissionScope,
        SocialTargetType,
    )
    from src.agency.social_gate import SocialSendGate
    from src.agency.store import FileSocialPermissionStore
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    store.add(
        SocialPermissionGrant(
            id="grant-devchat",
            target_id="@devchat",
            target_type=SocialTargetType.GROUP,
            scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
            allowed_topics=("runtime",),
        )
    )
    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        social_send_gate=SocialSendGate(store=store),
    )

    result = await skill.execute(
        "/telegram_send @devchat | Напишу без реального повода.",
        _ctx(
            approved=True,
            metadata={
                "social_send_candidate": True,
                "social_send_topic": "runtime",
                "social_send_addressed_to_zhvusha": False,
                "social_send_has_value_to_add": True,
            },
        ),
    )

    assert result.success is False
    assert "social_judgement_read_only" in result.response
    assert runtime.created == []


@pytest.mark.asyncio
async def test_autonomous_social_send_candidate_invalid_metadata_blocks_before_runtime_job(
    tmp_path: Path,
) -> None:
    from src.agency.social_gate import SocialSendGate
    from src.agency.store import FileSocialPermissionStore
    from src.skills.telegram_mcp_personal.skill import TelegramMCPPersonalSkill

    runtime = _Runtime()
    skill = TelegramMCPPersonalSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        actions_profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        social_send_gate=SocialSendGate(
            store=FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
        ),
    )

    result = await skill.execute(
        "/telegram_send @devchat | Некорректная metadata.",
        _ctx(
            approved=True,
            metadata={
                "social_send_candidate": True,
                "social_send_required_scope": "unknown_scope",
            },
        ),
    )

    assert result.success is False
    assert "invalid_social_send_request" in result.response
    assert runtime.created == []
