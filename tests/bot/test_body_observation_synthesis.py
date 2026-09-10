from __future__ import annotations

from typing import Any

import pytest
from src.bot import main as bot_main
from src.skills.base import AgentContext, SkillResult


class _ChatSkill:
    name = "chat_response"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        self.calls.append((message, context.metadata))
        return SkillResult(success=True, response="Живой ответ Жвуши.")


class _ChatSkillWithResponse:
    name = "chat_response"

    def __init__(self, response: str) -> None:
        self.response = response

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        return SkillResult(success=True, response=self.response)


class _ChatSkillWithResponses:
    name = "chat_response"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        self.calls.append((message, context.metadata))
        response = self.responses.pop(0)
        return SkillResult(success=True, response=response)


@pytest.mark.asyncio
async def test_body_observation_is_synthesized_by_chat_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={
            "prefer_chat_response_only": True,
            "chat_context_budget": "compressed",
            "chat_context_budget_reason": "compressed:short_personal_turn",
        },
    )
    body_result = SkillResult(
        success=True,
        response="Не хватает @username/id для Тоше.",
        metadata={
            "skill_name": "telegram_mcp_personal",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "missing_required_input",
                "pending_decision": {
                    "kind": "missing_required_input",
                    "missing_fields": ["chat_id"],
                },
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Пиши ему",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.response == "Живой ответ Жвуши."
    assert chat.calls[0][0] == "Пиши ему"
    metadata = chat.calls[0][1]
    assert metadata["suppress_memory_proposals"] is True
    assert metadata["prefer_chat_response_only"] is True
    assert metadata["disable_side_effect_intercepts"] is True
    assert metadata["disable_computer_use_tool"] is True
    assert metadata["disable_computer_use_intercept"] is True
    assert metadata["chat_context_budget"] == "focused"
    assert (
        metadata["chat_context_budget_reason"] == "body_observation:focused_observation"
    )
    assert "<BODY_OBSERVATION>" not in metadata["body_observation"]
    assert "missing_required_input" in metadata["body_observation"]


@pytest.mark.asyncio
async def test_body_observation_synthesis_uses_approved_original_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="External skill runtime completed; Жвуша собирает ответ.",
        metadata={
            "skill_name": "external_skill_runtime",
            "requires_zhvusha_response": True,
            "body_observation_synthesis_message": (
                "/external_skill_readonly source-grounding | Trace handler"
            ),
            "body_observation": {
                "event": "external_skill_runtime_completed",
                "external_skill_id": "source-grounding",
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        bot_main._body_observation_synthesis_message("разрешаю", body_result),
        context,
        body_result,
    )

    assert synthesized is not None
    assert (
        chat.calls[0][0] == "/external_skill_readonly source-grounding | Trace handler"
    )
    metadata = chat.calls[0][1]
    assert metadata["force_context_budget"] == "full"
    assert (
        metadata["chat_context_budget_reason"]
        == "body_observation:approved_original_message"
    )
    assert "chat_context_budget" not in metadata
    assert "prefer_chat_response_only" not in metadata


@pytest.mark.asyncio
async def test_computer_use_body_observation_uses_reasoning_loop_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={
            "prefer_chat_response_only": True,
            "chat_context_budget": "focused",
            "dialogue_context": "old state says no profile was found",
            "recent_decision_messages": "assistant asked user for Dotabuff link",
            "recent_messages": "old failed attempt asked for SteamID",
        },
    )
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_interactive_task",
                "selected_url": "https://www.google.com/search?q=kereexa+dotabuff",
                "summary": "Computer-use action completed.",
                "processed_context": (
                    "# Computer-use action\n"
                    "- status: completed\n"
                    "# Page state\n"
                    "- title: Google Search\n"
                    "- interactive_elements:\n"
                    "  - a a: DOTABUFF - Dota 2 Statistics"
                ),
                "instruction": (
                    "Если исходная цель ещё не достигнута, выведи следующую "
                    "команду /computer_use отдельной строкой."
                ),
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Проанализируй kereexa игрока в доте",
        context,
        body_result,
    )

    assert synthesized is not None
    metadata = chat.calls[0][1]
    assert metadata["force_context_budget"] == "full"
    assert metadata["body_observation_reasoning_continuation"] is True
    assert (
        metadata["chat_context_budget_reason"]
        == "body_observation:computer_use_reasoning_loop"
    )
    assert "prefer_chat_response_only" not in metadata
    assert "chat_context_budget" not in metadata
    assert "dialogue_context" not in metadata
    assert "recent_decision_messages" not in metadata
    assert "recent_messages" not in metadata


@pytest.mark.asyncio
async def test_computer_use_completed_result_uses_focused_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={
            "dialogue_context": "old state says no profile was found",
            "recent_messages": "old failed attempt asked for SteamID",
        },
    )
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_interactive_task",
                "summary": "Computer-use action completed.",
                "sources": ["https://www.dotabuff.com/players/997362076"],
                "processed_context": (
                    "# Computer-use action\n"
                    "- status: completed\n"
                    "- current_url: https://www.dotabuff.com/players/997362076\n"
                    "- title: Kereexa - Overview - DOTABUFF - Dota 2 Stats\n"
                    "1. result_detected: "
                    "https://www.dotabuff.com/players/997362076\n"
                    "# Page state\n"
                    "- interactive_elements:\n"
                    "  - a a: Matches"
                ),
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Проанализируй kereexa игрока в доте",
        context,
        body_result,
    )

    assert synthesized is not None
    metadata = chat.calls[0][1]
    assert metadata["prefer_chat_response_only"] is True
    assert metadata["disable_side_effect_intercepts"] is True
    assert metadata["disable_computer_use_tool"] is True
    assert metadata["disable_computer_use_intercept"] is True
    assert metadata["chat_context_budget"] == "focused"
    assert (
        metadata["chat_context_budget_reason"] == "body_observation:focused_observation"
    )
    assert "body_observation_reasoning_continuation" not in metadata
    assert "dialogue_context" not in metadata
    assert "recent_messages" not in metadata


@pytest.mark.asyncio
async def test_handle_skill_result_response_recurses_for_body_observation_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_body = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_interactive_task",
                "summary": "Intermediate observation.",
            },
        },
    )
    second_body = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_click",
                "summary": "Clicked next result.",
            },
        },
    )
    final = SkillResult(
        success=True,
        response="Финальный разбор после второго наблюдения.",
        metadata={"skill_name": "chat_response"},
    )
    calls: list[SkillResult] = []

    async def synthesize(
        _text: str,
        _context: AgentContext,
        result: SkillResult,
    ) -> SkillResult:
        calls.append(result)
        return second_body if len(calls) == 1 else final

    monkeypatch.setattr(bot_main, "_synthesize_body_observation_response", synthesize)
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={"return_response_text": True},
    )

    response = await bot_main._handle_skill_result_response(
        "Проанализируй игрока",
        context,
        first_body,
    )

    assert response == "Финальный разбор после второго наблюдения."
    assert calls == [first_body, second_body]


@pytest.mark.asyncio
async def test_computer_use_synthesis_retries_unverified_and_contradictory_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkillWithResponses(
        [
            (
                "точный профиль не пробился, вот OpenDota: "
                "https://www.opendota.com/players/349470236"
            ),
            (
                "нашла профиль: https://www.dotabuff.com/players/997362076\n"
                "по фактам Dotabuff: 75% core, Shadow Fiend 207 игр, "
                "Phantom Assassin 201 игра."
            ),
        ]
    )
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={"return_response_text": True},
    )
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_interactive_task",
                "summary": "Computer-use action completed.",
                "sources": ["https://www.dotabuff.com/players/997362076"],
                "processed_context": (
                    "# Computer-use action\n"
                    "- status: completed\n"
                    "- current_url: https://www.dotabuff.com/players/997362076\n"
                    "- title: Kereexa - Overview - DOTABUFF - Dota 2 Stats\n"
                    "- result_sections:\n"
                    "  - section 1: 75% CORE MOST PLAYED HEROES "
                    "Shadow Fiend 207 Phantom Assassin 201\n"
                    "1. clicked_profile: kereexa -> "
                    "https://www.dotabuff.com/players/997362076\n"
                    "2. result_detected: "
                    "https://www.dotabuff.com/players/997362076"
                ),
            },
        },
    )

    response = await bot_main._handle_skill_result_response(
        "Проанализируй kereexa игрока в доте",
        context,
        body_result,
    )

    assert response is not None
    assert "dotabuff.com/players/997362076" in response
    assert "opendota.com" not in response
    assert len(chat.calls) == 2
    assert "Предыдущий черновой ответ противоречил" in chat.calls[1][0]
    assert chat.calls[1][1]["body_observation_grounding_retry"] is True


@pytest.mark.asyncio
async def test_body_observation_synthesis_forces_full_for_grounding_failure_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        metadata={
            "prefer_chat_response_only": True,
            "chat_context_budget": "compressed",
            "chat_context_budget_reason": "compressed:short_personal_turn",
        },
    )
    body_result = SkillResult(
        success=False,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "body_observation_synthesis_message": (
                "Read-only web research не дал проверенных источников. "
                "Не отвечай из памяти."
            ),
            "body_observation": {
                "event": "web_research_completed",
                "sources": [],
                "summary": "не нашла URL для read-only web research.",
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        bot_main._body_observation_synthesis_message("разрешаю", body_result),
        context,
        body_result,
    )

    assert synthesized is not None
    metadata = chat.calls[0][1]
    assert metadata["force_context_budget"] == "full"
    assert (
        metadata["chat_context_budget_reason"]
        == "body_observation:approved_original_message"
    )
    assert "chat_context_budget" not in metadata
    assert "prefer_chat_response_only" not in metadata
    assert '"sources": []' in metadata["body_observation"]


@pytest.mark.asyncio
async def test_web_research_synthesis_rejects_urls_outside_body_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkillWithResponse(
        "Подтверждено: https://www.python.org/downloads/release/python-3145/"
    )
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "web_research_completed",
                "sources": [
                    "https://docs.python.org/3/whatsnew/3.14.html",
                ],
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Что с Python 3.14?",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is False
    assert "не показываю как подтверждённый" in synthesized.response
    assert synthesized.metadata["body_observation_grounding_rejected"] is True
    assert synthesized.metadata["unverified_urls"] == (
        "https://www.python.org/downloads/release/python-3145",
    )


@pytest.mark.asyncio
async def test_web_research_synthesis_allows_markdown_wrapped_body_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkillWithResponse(
        "Подтверждено: `https://example.com/` читается как Example Domain."
    )
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "web_research_completed",
                "sources": ["https://example.com/"],
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Открой example.com",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is True
    assert synthesized.response == (
        "Подтверждено: `https://example.com/` читается как Example Domain."
    )
    assert "body_observation_grounding_rejected" not in synthesized.metadata


@pytest.mark.asyncio
async def test_web_research_synthesis_rewrites_artifact_only_grounding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "deliver_artifacts_to_chat": True,
            "artifacts": ("agent_runtime/browser_artifacts/screenshot-page.png",),
            "body_observation": {
                "event": "web_research_completed",
                "summary": (
                    "Скриншот источника сохранён, но текст страницы read-only "
                    "прочитать не удалось."
                ),
                "sources": ["https://ru.wikipedia.org/wiki/Special:Random"],
                "artifacts": [
                    "agent_runtime/browser_artifacts/screenshot-page.png",
                ],
                "artifact_only": True,
                "readable_source_count": 0,
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Открой статью и сделай скриншот",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is True
    assert "Скриншот сохранила" in synthesized.response
    assert "текст страницы read-only прочитать не удалось" in synthesized.response
    assert "содержание страницы не подтверждаю" in synthesized.response
    assert "agent_runtime/browser_artifacts/screenshot-page.png" in synthesized.response
    assert chat.calls == []
    assert synthesized.metadata["body_observation_grounding_rewritten"] is True


@pytest.mark.asyncio
async def test_web_research_synthesis_requests_human_verification_for_security_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=False,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "deliver_artifacts_to_chat": True,
            "body_observation": {
                "event": "web_research_completed",
                "query": "https://www.dotabuff.com/search?utf8=%E2%9C%93&q=kereexa",
                "summary": (
                    "Сайт показал security verification/challenge; целевую "
                    "страницу и скриншот read-only получить не удалось."
                ),
                "sources": [],
                "artifacts": [],
                "findings": [
                    {
                        "claim": (
                            "Источник открыл anti-bot/security verification "
                            "вместо целевой страницы"
                        ),
                        "status": "rejected",
                    }
                ],
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Открой Dotabuff и сделай скрин статистики",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is False
    assert "Пройди проверку" in synthesized.response
    assert "окне браузера, которым я управляю" in synthesized.response
    assert "открывать ссылку отдельно не нужно" in synthesized.response
    assert "продолжу сама" in synthesized.response
    assert "готово" not in synthesized.response.lower()
    assert "dotabuff.com/search" in synthesized.response
    assert chat.calls == []
    assert synthesized.metadata["body_observation_human_verification_required"] is True
    assert synthesized.metadata["requires_user_action"] is True
    assert synthesized.metadata["dialogue_state_patch"] == {
        "pending_action": "human_verification_challenge",
        "selected_skill": "web_research",
        "last_result": "requires_human_verification",
    }


@pytest.mark.asyncio
async def test_web_research_synthesis_keeps_readable_sources_with_partial_verification_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkillWithResponse(
        "По прочитанным источникам: STRATZ и OpenDota открылись, Dotabuff упёрся "
        "в security verification."
    )
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "web_research",
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "web_research_completed",
                "query": "kereexa Dota 2 Dotabuff OpenDota STRATZ SteamID",
                "summary": "Прочитала 4 источник(а) read-only.",
                "sources": [
                    "https://stratz.com/",
                    "https://www.opendota.com/",
                    "https://www.dotabuff.com/",
                ],
                "readable_source_count": 4,
                "processed_context": (
                    "Источник прочитан через browser_read_url: https://stratz.com/\n"
                    "Источник прочитан через browser_read_url: https://www.opendota.com/\n"
                    "Источник найден, но не прочитан через browser_read_url: "
                    "https://www.dotabuff.com/ (browser reached security verification "
                    "challenge for www.dotabuff.com)"
                ),
                "findings": [
                    {
                        "claim": (
                            "Источник открыл anti-bot/security verification вместо "
                            "целевой страницы: https://www.dotabuff.com/"
                        ),
                        "status": "rejected",
                    },
                    {
                        "claim": (
                            "Источник прочитан через browser_read_url: "
                            "https://stratz.com/"
                        ),
                        "status": "supported",
                    },
                ],
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Проанализируй kereexa",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is True
    assert "STRATZ и OpenDota открылись" in synthesized.response
    assert "body_observation_verification_blocked" not in synthesized.metadata


@pytest.mark.asyncio
async def test_computer_use_synthesis_requests_human_verification_for_captcha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _ChatSkill()
    monkeypatch.setattr(bot_main, "_skills", [chat])
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")
    body_result = SkillResult(
        success=True,
        response="",
        metadata={
            "skill_name": "computer_use",
            "requires_zhvusha_response": True,
            "deliver_artifacts_to_chat": True,
            "artifacts": ("agent_runtime/computer_use/screenshots/captcha.png",),
            "body_observation": {
                "event": "computer_use_action_completed",
                "selected_action": "browser_interactive_task",
                "selected_url": "https://steamcommunity.com/search/users/#text=kereexa",
                "summary": "Страница показала captcha/security challenge.",
                "processed_context": (
                    "Unfortunately, bots use this search too. Please complete "
                    "the following challenge to confirm this search was made by "
                    "a human."
                ),
                "artifacts": [
                    "agent_runtime/computer_use/screenshots/captcha.png",
                ],
            },
        },
    )

    synthesized = await bot_main._synthesize_body_observation_response(
        "Проанализируй kereexa",
        context,
        body_result,
    )

    assert synthesized is not None
    assert synthesized.success is False
    assert "Пройди проверку" in synthesized.response
    assert "открывать ссылку отдельно не нужно" in synthesized.response
    assert "продолжу сама" in synthesized.response
    assert "готово" not in synthesized.response.lower()
    assert "steamcommunity.com/search" in synthesized.response
    assert chat.calls == []
    assert synthesized.metadata["body_observation_human_verification_required"] is True
    assert synthesized.metadata["requires_user_action"] is True
    assert synthesized.metadata["deliver_artifacts_to_chat"] is True
    assert synthesized.metadata["artifacts"] == (
        "agent_runtime/computer_use/screenshots/captcha.png",
    )
    assert synthesized.metadata["source_skill_name"] == "computer_use"
    assert synthesized.metadata["dialogue_state_patch"] == {
        "pending_action": "human_verification_challenge",
        "selected_skill": "computer_use",
        "last_result": "requires_human_verification",
    }
