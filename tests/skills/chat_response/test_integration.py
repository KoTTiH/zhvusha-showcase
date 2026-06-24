"""Integration tests for ChatResponseSkill.

Ported from tests/test_chat_response_skill.py in phase 7.4. Contexts use the
v4 ``AgentContext`` frozen dataclass; ``LLMRouter`` is patched at the module
level the same way the legacy tests did.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.protocols import LLMResponse, LLMToolResponse, LLMUsage
from src.skills.base import AgentContext
from src.skills.chat_response.skill import ChatResponseSkill

if TYPE_CHECKING:
    from pathlib import Path

    from src.knowledge import KnowledgeStore

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"
_PATCH_ROUTER = "src.skills.chat_response.skill.get_router"
_PATCH_PEOPLE = "src.skills.chat_response.skill.get_people_manager"


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


@dataclass(frozen=True)
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


def _tool_resp(text: str) -> LLMToolResponse:
    return LLMToolResponse(
        content_blocks=[_TextBlock(text)],
        stop_reason="end_turn",
        model="haiku",
        usage=LLMUsage(),
    )


def _settings(tmp_path: str = "test_ws") -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path=tmp_path,
        claude_cli_path="claude",
        public_info_about_nikita="Nikita is a developer.",
        admin_user_id=12345,
        channel_id="@test",
        chat_assistant_tier="analyst",
        chat_agentic_timeout_seconds=300.0,
    )


def _context(mode: str = "personal", user_id: int = 12345) -> AgentContext:
    return AgentContext(
        user_id=user_id,
        chat_id=user_id,
        mode=mode,  # type: ignore[arg-type]
        message_id=1,
        bot=None,
    )


def _setup_workspace(root: str) -> None:
    from pathlib import Path

    ws = Path(root)
    (ws / "personality").mkdir(parents=True, exist_ok=True)
    (ws / "personality" / "core.md").write_text("I am Zhvusha.")
    (ws / "personality" / "genes.md").write_text("Curiosity: HIGH")
    (ws / "diary").mkdir(parents=True, exist_ok=True)
    (ws / "memory" / "people").mkdir(parents=True, exist_ok=True)


def _mock_people(*, interaction_count: int = 0) -> MagicMock:
    mgr = MagicMock()
    mgr.get_or_create_profile = MagicMock(return_value={"user_id": 12345})
    mgr.record_interaction = MagicMock(return_value=False)
    mgr.get_profile_for_context = MagicMock(return_value="")
    mgr.get_interaction_count = MagicMock(return_value=interaction_count)
    return mgr


class TestCanHandle:
    async def test_plain_text(self) -> None:
        skill = ChatResponseSkill()
        ctx = _context()
        assert await skill.can_handle("привет", ctx) == 0.3

    async def test_ignores_commands(self) -> None:
        skill = ChatResponseSkill()
        ctx = _context()
        assert await skill.can_handle("/post hello", ctx) == 0.0
        assert await skill.can_handle("/kwork", ctx) == 0.0
        assert await skill.can_handle("/morning", ctx) == 0.0


def test_build_user_prompt_includes_dialogue_state_before_full_history() -> None:
    prompt = ChatResponseSkill._build_user_prompt(
        "Пиши ему",
        dialogue_context=(
            "pending_action: telegram_send\n"
            "recipient_hint: Тоше\n"
            "executable_chat_id: missing"
        ),
        recent_messages=(
            "Собеседник: Не мне, а Тоше\nЖвуша: Не хватает @username/id для Тоше."
        ),
    )

    assert "<DIALOGUE_STATE>" in prompt
    assert "recipient_hint: Тоше" in prompt
    assert "<CONVERSATION_HISTORY>" in prompt
    assert prompt.index("<DIALOGUE_STATE>") < prompt.index("<CONVERSATION_HISTORY>")
    assert "<CURRENT_MESSAGE>\nПиши ему\n</CURRENT_MESSAGE>" in prompt


def test_build_user_prompt_includes_body_observation_before_current_message() -> None:
    prompt = ChatResponseSkill._build_user_prompt(
        "Пиши ему",
        body_observation='{"missing_fields": ["chat_id"]}',
    )

    assert "<BODY_OBSERVATION>" in prompt
    assert '"missing_fields": ["chat_id"]' in prompt
    assert prompt.index("<BODY_OBSERVATION>") < prompt.index("<CURRENT_MESSAGE>")


def test_body_observation_execution_not_attempted_is_rendered_as_routing_check() -> (
    None
):
    body_observation = json.dumps(
        {
            "event": "body_layer_check_completed",
            "status": "missing_fields",
            "routing": {
                "candidate_skill": "external_skill_acquisition",
                "decision": "needs_more_input",
            },
            "execution": {
                "attempted": False,
                "reason": "missing_required_fields",
            },
            "source_status": "absent",
            "sources": [],
            "artifacts": [],
            "side_effects": [],
            "missing_fields": ["target_source", "approval_scope"],
        },
        ensure_ascii=False,
    )
    prompt = ChatResponseSkill._build_user_prompt(
        "проверь и сделай",
        body_observation=body_observation,
    )

    policy = prompt.split("<BODY_OBSERVATION_POLICY>\n", maxsplit=1)[1].split(
        "\n</BODY_OBSERVATION_POLICY>",
        maxsplit=1,
    )[0]
    assert "execution.attempted=false" in policy
    assert "routing/safety/missing-fields checking" in policy
    assert "file reads" in policy
    assert "tool execution" in policy
    assert "repository verification" in policy
    assert "physical artifacts" in policy
    assert "completed side effects" in policy
    assert "source status" in policy
    assert "verified means confirmed/readable evidence" in policy
    assert "absent means empty sources/artifacts" in policy
    assert prompt.index("<BODY_OBSERVATION_POLICY>") < prompt.index(
        "<BODY_OBSERVATION>"
    )
    assert '"attempted": false' in prompt
    assert '"source_status": "absent"' in prompt
    assert '"missing_fields": [' in prompt


class TestExecute:
    async def test_personal_mode(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("Привет, Никита!"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("привет", _context(mode="personal"))

        assert result.success is True
        assert result.response == "Привет, Никита!"
        mock_router.generate.assert_awaited_once()

        request = mock_router.generate.call_args.args[0]
        system = request.system
        assert "I am Zhvusha." in system
        user_prompt = request.prompt
        assert "привет" in user_prompt

    async def test_social_mode(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("Хахаха"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("hello", _context(mode="social"))

        assert result.success is True
        request = mock_router.generate.call_args.args[0]
        system = request.system
        assert "Today was interesting" not in system

    async def test_assistant_mode(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(
            return_value=_llm_resp("Привет! Я помощник Никиты.")
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("привет", _context(mode="assistant"))

        assert result.success is True
        assert result.response == "Привет! Я помощник Никиты."

    async def test_assistant_metadata_can_limit_knowledge_context_to_categories(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        async def hybrid_search(query, *, category=None, limit=10, tags=None):
            del query, limit, tags
            if category == "research":
                return [SimpleNamespace(id=1, title="public", rrf_score=0.9)]
            if category == "intel.channels":
                return [SimpleNamespace(id=2, title="channel", rrf_score=0.8)]
            if category == "dev.private":
                return [SimpleNamespace(id=3, title="private", rrf_score=1.0)]
            return []

        knowledge_store = SimpleNamespace(
            hybrid_search=AsyncMock(side_effect=hybrid_search),
            get_summaries=AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id=1,
                        title="PUBLIC_RESEARCH_TITLE",
                        summary="PUBLIC_RESEARCH_SUMMARY",
                    ),
                    SimpleNamespace(
                        id=2,
                        title="PUBLIC_CHANNEL_TITLE",
                        summary="PUBLIC_CHANNEL_SUMMARY",
                    ),
                ]
            ),
        )
        skill = ChatResponseSkill(
            knowledge_store=cast("KnowledgeStore", knowledge_store)
        )
        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("ok"))
        context = AgentContext(
            user_id=999,
            chat_id=999,
            mode="assistant",
            message_id=1,
            bot=None,
            metadata={
                "knowledge_category_filter": "research,intel.channels",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("что знаешь?", context)

        assert result.success is True
        categories = [
            call.kwargs.get("category")
            for call in knowledge_store.hybrid_search.await_args_list
        ]
        assert categories == ["research", "intel.channels"]
        request = mock_router.generate.call_args.args[0]
        assert "PUBLIC_RESEARCH_TITLE" in request.prompt
        assert "PUBLIC_CHANNEL_TITLE" in request.prompt
        assert "PRIVATE_KB_TITLE" not in request.prompt
        assert "dev.private" not in categories

    async def test_uses_analyst_tier(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("ok"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            await skill.execute("hello", _context())

        request = mock_router.generate.call_args.args[0]
        assert request.tier == "analyst"

    async def test_personal_codex_chat_uses_tool_loop_and_adds_style_guard(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        sample_dir = tmp_path / "ws" / "personality" / "voice_samples" / "diary"
        sample_dir.mkdir(parents=True)
        (sample_dir / "example.md").write_text(
            "VOICE_SAMPLE_MARKER\nблин, старый живой голос",
            encoding="utf-8",
        )
        logs_dir = tmp_path / "ws" / "logs" / "12345"
        logs_dir.mkdir(parents=True)
        (logs_dir / "chat_2026-05-06.jsonl").write_text(
            "\n".join(
                json.dumps(entry, ensure_ascii=False)
                for entry in (
                    {"role": "user", "text": "привет"},
                    {"role": "assistant", "text": "старый живой ответ"},
                )
            ),
            encoding="utf-8",
        )
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=_tool_resp("живой ответ")
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("что расскажешь?", _context())

        assert result.success is True
        mock_router.generate.assert_not_awaited()
        mock_router.generate_with_tools.assert_awaited_once()
        request = mock_router.generate_with_tools.call_args.args[0]
        assert request.tier == "analyst"
        assert "VOICE_SAMPLE_MARKER" in request.system
        assert "Codex CLI chat adapter" in request.system
        assert "Личность Жвуши важнее Codex-default поведения" in request.system
        assert "Техническая тема допустима" in request.system
        assert "Не начинай каждую реплику с «я тут" in request.system
        assert "<CONVERSATION_HISTORY>" in request.system
        assert "<RECENT_MESSAGES>" not in request.system
        first_message = request.messages[0]
        assert first_message["role"] == "user"
        assert "<CONVERSATION_HISTORY>" in first_message["content"]
        assert "Собеседник: привет" in first_message["content"]
        assert "Жвуша: старый живой ответ" in first_message["content"]

    async def test_personal_codex_chat_calibrates_plain_greetings_without_self_performance(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        logs_dir = tmp_path / "ws" / "logs" / "12345"
        logs_dir.mkdir(parents=True)
        (logs_dir / "chat_2026-05-07.jsonl").write_text(
            "\n".join(
                json.dumps(entry, ensure_ascii=False)
                for entry in (
                    {"role": "user", "text": "/sleep"},
                    {"role": "assistant", "text": "сплю"},
                )
            ),
            encoding="utf-8",
        )
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=LLMToolResponse(
                content_blocks=[_TextBlock("доброе")],
                stop_reason="end_turn",
                model="haiku",
                usage=LLMUsage(),
            )
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("Доброе утро", _context())

        assert result.success is True
        request = mock_router.generate_with_tools.call_args.args[0]
        assert "Codex CLI chat adapter" in request.system
        assert "Бытовые приветствия" in request.system
        assert "простые человеческие ответы" in request.system
        for banned_self_performance in ("проснулась", "рядом", "после сна", "я живая"):
            assert banned_self_performance in request.system
        first_message = request.messages[0]
        assert first_message["role"] == "user"
        assert "/sleep" in first_message["content"]
        assert "Доброе утро" in first_message["content"]

    async def test_vscode_checkin_gets_interface_context_without_codex_probe_history(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        logs_dir = tmp_path / "ws" / "logs" / "vscode"
        logs_dir.mkdir(parents=True)
        (logs_dir / "chat_2026-05-20.jsonl").write_text(
            "\n".join(
                json.dumps(entry, ensure_ascii=False)
                for entry in (
                    {
                        "role": "user",
                        "source": "vscode",
                        "source_actor": "codex",
                        "text": "проверяет bridge",
                    },
                    {
                        "role": "assistant",
                        "source": "vscode",
                        "source_actor": "zhvusha",
                        "text": "Я в отдельном VS Code-чате.",
                    },
                    {
                        "role": "user",
                        "source": "vscode",
                        "source_actor": "user",
                        "text": "обычный человеческий вопрос",
                    },
                )
            ),
            encoding="utf-8",
        )
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("нормально"))
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "chat_log_id": "vscode",
                "interface": "vscode",
                "source_actor": "user",
                "interface_context": (
                    "Текущий канал контакта: VS Code chat через локальный bridge."
                ),
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("че как ты", context)

        assert result.success is True
        request = mock_router.generate.call_args.args[0]
        assert "обычный человеческий вопрос" in request.prompt
        assert "<INTERFACE_CONTEXT>" in request.prompt
        assert "Текущий канал контакта: VS Code chat" in request.prompt
        assert "Codex: проверяет bridge" not in request.prompt
        assert "Я в отдельном VS Code-чате." not in request.prompt

    async def test_context_budget_compressed_live_checkin_uses_identity_kernel(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        logs_dir = tmp_path / "ws" / "logs" / "vscode"
        logs_dir.mkdir(parents=True)
        (logs_dir / "chat_2026-05-20.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"role": "user", "source": "vscode", "text": f"старый {i}"},
                    ensure_ascii=False,
                )
                for i in range(8)
            ),
            encoding="utf-8",
        )
        settings = _settings(ws)
        planner = AsyncMock()
        planner.retrieve_for_question = AsyncMock()
        skill = ChatResponseSkill(decision_engine=planner)
        skill.set_manager_capability_summary("## Внутренний граф возможностей\nSECRET")

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("живой ответ"))

        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "interface": "vscode",
                "chat_log_id": "vscode",
                "interface_context": "Текущий канал контакта: VS Code chat.",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("как дела у тебя?", context)

        assert result.success is True
        planner.retrieve_for_question.assert_not_awaited()
        mock_router.generate_with_tools.assert_not_awaited()
        request = mock_router.generate.call_args.args[0]
        assert request.tier == "worker"
        assert "version: zhvusha-identity-kernel-v1" in request.system
        assert "Curiosity: HIGH" not in request.system
        assert "Внутренний граф возможностей" not in request.system
        assert "SECRET" not in request.system
        assert "## Твои инструменты" not in request.system
        assert "compressed/focused режиме" in request.system
        assert "<CURRENT_LINE_SUMMARY>" in request.prompt
        assert "без отчёта о готовности" in request.prompt
        assert "старый 1" not in request.prompt
        assert "старый 7" in request.prompt

    async def test_context_budget_focused_runtime_question_skips_heavy_paths(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        planner = AsyncMock()
        planner.retrieve_for_question = AsyncMock()
        skill = ChatResponseSkill(decision_engine=planner)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("по делу"))
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "interface": "vscode",
                "chat_log_id": "vscode",
                "interface_context": "VS Code bridge работает через локальный HTTP.",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute(
                "почему VS Code bridge так долго отвечает?",
                context,
            )

        assert result.success is True
        planner.retrieve_for_question.assert_not_awaited()
        mock_router.generate_with_tools.assert_not_awaited()
        request = mock_router.generate.call_args.args[0]
        assert request.tier == "worker"
        assert "Никита обсуждает техническое поведение чата" in request.prompt
        assert "<INTERFACE_CONTEXT>" in request.prompt
        assert "VS Code bridge работает" in request.prompt

    async def test_context_budget_pending_dialogue_forces_full_context(
        self,
        tmp_path: Path,
    ) -> None:
        from src.core.decision import RetrievalResult

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        planner = AsyncMock()
        planner.retrieve_for_question = AsyncMock(
            return_value=RetrievalResult(memory_context="незавершённая отправка")
        )
        skill = ChatResponseSkill(decision_engine=planner)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("продолжаю задачу"))
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=settings.admin_user_id,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "telegram",
                "interface": "telegram",
                "dialogue_context": "pending_action: telegram_send",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("ну что?", context)

        assert result.success is True
        planner.retrieve_for_question.assert_awaited_once()
        request = mock_router.generate.call_args.args[0]
        assert request.tier == "analyst"
        assert "Curiosity: HIGH" in request.system
        assert "незавершённая отправка" in request.prompt

    async def test_context_budget_implicit_tool_intent_forces_full_context(
        self,
        tmp_path: Path,
    ) -> None:
        from src.core.decision import RetrievalResult

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        planner = AsyncMock()
        planner.retrieve_for_question = AsyncMock(
            return_value=RetrievalResult(memory_context="нужны последние сообщения")
        )
        skill = ChatResponseSkill(decision_engine=planner)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("смотрю контекст"))
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=settings.admin_user_id,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={"source": "telegram", "interface": "telegram"},
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("посмотри последние от Тоши", context)

        assert result.success is True
        planner.retrieve_for_question.assert_awaited_once()
        request = mock_router.generate.call_args.args[0]
        assert request.tier == "analyst"
        assert "Curiosity: HIGH" in request.system
        assert "нужны последние сообщения" in request.prompt

    async def test_context_budget_external_skill_gap_intent_forces_full_context(
        self,
        tmp_path: Path,
    ) -> None:
        from src.core.decision import RetrievalResult

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        planner = AsyncMock()
        planner.retrieve_for_question = AsyncMock(
            return_value=RetrievalResult(memory_context="нужен Kubernetes debug skill")
        )
        skill = ChatResponseSkill(decision_engine=planner)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("нужен навык"))
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "interface": "vscode",
                "chat_log_id": "vscode",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute(
                "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
                context,
            )

        assert result.success is True
        planner.retrieve_for_question.assert_awaited_once()
        request = mock_router.generate.call_args.args[0]
        assert request.tier == "analyst"
        assert "Curiosity: HIGH" in request.system
        assert "нужен Kubernetes debug skill" in request.prompt

    async def test_personal_codex_chat_contract_allows_technical_experience_without_readiness_report(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=_tool_resp("нормально")
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "chat_log_id": "vscode",
                "interface": "vscode",
                "interface_context": (
                    "Текущий канал контакта: VS Code chat через локальный bridge."
                ),
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("как дела у тебя?", context)

        assert result.success is True
        request = mock_router.generate_with_tools.call_args.args[0]
        assert "техническое тело" in request.system
        assert "Технический контекст можно использовать" in request.system
        assert "не меню доступности" in request.system
        assert "не выноси в ответ внутреннее правило" in request.system
        assert "просто отвечай естественно" in request.system
        assert "<INTERFACE_CONTEXT>" in request.messages[0]["content"]

    async def test_personal_agentic_timeout_comes_from_settings(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        settings.chat_agentic_timeout_seconds = 123.0
        skill = ChatResponseSkill()

        captured: dict[str, float] = {}

        async def fake_wait_for(awaitable: Any, timeout: float) -> Any:
            captured["timeout"] = timeout
            return await awaitable

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=_tool_resp("живой ответ")
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
            patch(
                "src.skills.chat_response.skill.asyncio.wait_for",
                side_effect=fake_wait_for,
            ),
        ):
            result = await skill.execute("что делаешь?", _context())

        assert result.success is True
        assert captured["timeout"] == 123.0

    async def test_computer_use_intercept_returns_body_observation_to_dispatcher(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        async def side_effect_invoker(
            command: str,
            context: AgentContext,
        ) -> Any:
            assert command.startswith("/computer_use ")
            assert context.mode == "personal"
            from src.skills.base import SkillResult

            return SkillResult(
                success=True,
                response="",
                metadata={
                    "skill_name": "computer_use",
                    "requires_zhvusha_response": True,
                    "deliver_artifacts_to_chat": True,
                    "artifacts": (
                        "agent_runtime/computer_use/screenshots/lower-page.png",
                    ),
                    "body_observation": {
                        "event": "computer_use_action_completed",
                        "selected_action": "browser_scroll",
                        "status": "completed",
                        "artifacts": [
                            "agent_runtime/computer_use/screenshots/lower-page.png",
                        ],
                    },
                },
            )

        skill = ChatResponseSkill(side_effect_invoker=side_effect_invoker)
        skill.set_manager_capability_summary(
            "## Внутренний граф возможностей\n"
            "- agent_profile.computer_use.active_gui: available"
        )
        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=_tool_resp(
                "щас пролистаю\n"
                '/computer_use {"action":"browser_scroll",'
                '"metadata":{"capture_screenshot":"true"}}'
            )
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute(
                "снизу есть ещё интерпретации, заскринь их тоже",
                _context(),
            )

        assert result.metadata["skill_name"] == "computer_use"
        assert result.metadata["requires_zhvusha_response"] is True
        assert (
            result.metadata["body_observation"]["selected_action"] == "browser_scroll"
        )
        assert result.metadata["artifacts"] == (
            "agent_runtime/computer_use/screenshots/lower-page.png",
        )
        assert "computer-use observation получен" not in result.response

    async def test_agentic_loop_uses_structured_computer_use_tool(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        captured: dict[str, Any] = {}

        async def side_effect_invoker(
            command: str,
            context: AgentContext,
        ) -> Any:
            assert command.startswith("/computer_use ")
            captured["payload"] = json.loads(command.removeprefix("/computer_use "))
            captured["context"] = context
            from src.skills.base import SkillResult

            return SkillResult(
                success=True,
                response="",
                metadata={
                    "skill_name": "computer_use",
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "computer_use_action_completed",
                        "selected_action": "browser_click",
                        "status": "completed",
                    },
                },
            )

        skill = ChatResponseSkill(side_effect_invoker=side_effect_invoker)
        skill.set_manager_capability_summary(
            "## Внутренний граф возможностей\n"
            "- agent_profile.computer_use.active_gui: available"
        )
        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=LLMToolResponse(
                content_blocks=[
                    _ToolUseBlock(
                        id="computer_call_1",
                        name="computer_use",
                        input={
                            "action": "browser_click",
                            "target": "DOTABUFF - Dota 2 Statistics",
                            "goal": "открыть профильный результат",
                            "metadata": {"capture_screenshot": "true"},
                        },
                    )
                ],
                stop_reason="tool_use",
                model="haiku",
                usage=LLMUsage(),
            )
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)
        context = _context()

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("продолжай browser task", context)

        assert result.metadata["skill_name"] == "computer_use"
        assert result.metadata["requires_zhvusha_response"] is True
        assert captured["context"] == context
        assert captured["payload"] == {
            "action": "browser_click",
            "goal": "открыть профильный результат",
            "metadata": {"capture_screenshot": "true"},
            "target": "DOTABUFF - Dota 2 Statistics",
        }
        first_request = mock_router.generate_with_tools.call_args.args[0]
        tool_names = {tool.name for tool in first_request.tools}
        assert "computer_use" in tool_names
        assert "structured tool `computer_use`" in first_request.system
        assert '/computer_use {"action"' not in first_request.system

    async def test_agentic_loop_retries_legacy_computer_use_text_as_tool_call(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        captured: dict[str, Any] = {}

        async def side_effect_invoker(
            command: str,
            context: AgentContext,
        ) -> Any:
            captured["payload"] = json.loads(command.removeprefix("/computer_use "))
            captured["context"] = context
            from src.skills.base import SkillResult

            return SkillResult(
                success=True,
                response="",
                metadata={
                    "skill_name": "computer_use",
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "computer_use_action_completed",
                        "selected_action": "browser_status",
                        "status": "completed",
                    },
                },
            )

        skill = ChatResponseSkill(side_effect_invoker=side_effect_invoker)
        skill.set_manager_capability_summary(
            "## Внутренний граф возможностей\n"
            "- agent_profile.computer_use.active_gui: available"
        )
        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            side_effect=[
                _tool_resp(
                    "internal computer_use не стартанул: нужна команда формата "
                    '/computer_use {"action":"browser_status"}'
                ),
                LLMToolResponse(
                    content_blocks=[
                        _ToolUseBlock(
                            id="computer_call_1",
                            name="computer_use",
                            input={"action": "browser_status"},
                        )
                    ],
                    stop_reason="tool_use",
                    model="haiku",
                    usage=LLMUsage(),
                ),
            ]
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)
        context = _context()

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("проверь живой браузер", context)

        assert result.metadata["skill_name"] == "computer_use"
        assert captured["context"] == context
        assert captured["payload"] == {"action": "browser_status"}
        assert mock_router.generate_with_tools.await_count == 2

    async def test_agentic_loop_serializes_codex_tool_blocks_for_project_tools(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "README.md").write_text("project evidence", encoding="utf-8")
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            side_effect=[
                LLMToolResponse(
                    content_blocks=[
                        _ToolUseBlock(
                            id="call_1",
                            name="read_project_file",
                            input={"path": "README.md"},
                        )
                    ],
                    stop_reason="tool_use",
                    model="haiku",
                    usage=LLMUsage(),
                ),
                _tool_resp("нашла файл"),
            ]
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "chat_log_id": "vscode",
                "interface": "vscode",
                "interface_context": "VS Code chat",
                "project_root": str(project_root),
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("прочитай README проекта", context)

        assert result.success is True
        assert result.response == "нашла файл"
        assert mock_router.generate_with_tools.await_count == 2
        second_request = mock_router.generate_with_tools.call_args_list[1].args[0]
        json.dumps(second_request.messages)
        assistant_blocks = second_request.messages[1]["content"]
        assert assistant_blocks == [
            {
                "id": "call_1",
                "name": "read_project_file",
                "input": {"path": "README.md"},
                "type": "tool_use",
            }
        ]
        tool_results = second_request.messages[2]["content"]
        assert tool_results[0]["tool_use_id"] == "call_1"
        assert "project evidence" in tool_results[0]["content"]

    async def test_agentic_loop_passes_vscode_codex_metadata_to_operator_tools(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            side_effect=[
                LLMToolResponse(
                    content_blocks=[
                        _ToolUseBlock(
                            id="call_rollback",
                            name="rollback_morning_personality_tail",
                            input={
                                "approval_id": ("job-c2d0b8d08a084b07808454272612ec5c"),
                                "dry_run": True,
                            },
                        )
                    ],
                    stop_reason="tool_use",
                    model="haiku",
                    usage=LLMUsage(),
                ),
                _tool_resp("готово"),
            ]
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)
        context = AgentContext(
            user_id=settings.admin_user_id,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "source_actor": "codex",
                "chat_log_id": "vscode",
                "interface": "vscode",
                "force_context_budget": "full",
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("rollback dry-run", context)

        assert result.success is True
        first_request = mock_router.generate_with_tools.call_args_list[0].args[0]
        assert "rollback_morning_personality_tail" in {
            tool.name for tool in first_request.tools
        }
        second_request = mock_router.generate_with_tools.call_args_list[1].args[0]
        tool_results = second_request.messages[2]["content"]
        assert tool_results[0]["tool_use_id"] == "call_rollback"
        assert "dry-run morning rollback" in tool_results[0]["content"]

    async def test_episodic_record_failure_does_not_block_chat_response(
        self,
        tmp_path: Path,
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        episodic = AsyncMock()
        episodic.record = AsyncMock(side_effect=RuntimeError("embeddings down"))
        skill = ChatResponseSkill(episodic=episodic)

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=_tool_resp("отвечаю без памяти")
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("проверь проект", _context())

        assert result.success is True
        assert result.response == "отвечаю без памяти"
        assert episodic.record.await_count == 2

    async def test_agentic_progress_loop_sends_and_cleans_status(self) -> None:
        from src.skills.chat_response.skill import _agentic_progress_loop

        bot = SimpleNamespace(
            send_chat_action=AsyncMock(),
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=77)),
            edit_message_text=AsyncMock(),
            delete_message=AsyncMock(),
        )

        task = asyncio.create_task(
            _agentic_progress_loop(
                bot=bot,
                chat_id=12345,
                initial_delay=0.0,
                update_interval=0.01,
                typing_interval=0.01,
            )
        )
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        bot.send_chat_action.assert_awaited()
        bot.send_message.assert_awaited_once()
        bot.delete_message.assert_awaited_once_with(chat_id=12345, message_id=77)

    async def test_zhvusha_identity_anchor_pins_gender_and_living_voice_across_modes(
        self, tmp_path: Path
    ) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        anchor_markers = (
            "Жвуша говорит о себе в женском грамматическом роде",
            "не зеркаль мужской род",
            "«прочитал»",
            "«смотрел»",
            "Живость остаётся через конкретную позицию",
            "техническое тело Жвуши",
            "Технический контекст можно использовать",
            "человеческий вопрос нельзя подменять подтверждением готовности",
        )

        with patch(_PATCH_SETTINGS, return_value=settings):
            systems = [
                skill._build_system(
                    "personal",
                    personality_context="I am Zhvusha.",
                    public_info=settings.public_info_about_nikita,
                    interaction_count=0,
                    people_context="Никита",
                    current_user_id=settings.admin_user_id,
                ),
                skill._build_system(
                    "assistant",
                    personality_context="I am Zhvusha.",
                    public_info=settings.public_info_about_nikita,
                    interaction_count=3,
                    current_user_id=777,
                ),
                skill._build_system(
                    "social",
                    personality_context="I am Zhvusha.",
                    public_info=settings.public_info_about_nikita,
                    interaction_count=0,
                    current_user_id=777,
                ),
            ]

        for system in systems:
            for marker in anchor_markers:
                assert marker in system

        mock_router = AsyncMock()
        mock_router.get_adapter = MagicMock(
            return_value=SimpleNamespace(name="codex_cli")
        )
        mock_router.generate_with_tools = AsyncMock(
            return_value=LLMToolResponse(
                content_blocks=[_TextBlock("прочитала канал")],
                stop_reason="end_turn",
                model="haiku",
                usage=LLMUsage(),
            )
        )
        mock_router.generate = AsyncMock(side_effect=AssertionError)

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("Как ты прочитал канал?", _context())

        assert result.success is True
        request = mock_router.generate_with_tools.call_args.args[0]
        assert request.tier == "analyst"
        assert "Codex CLI chat adapter" in request.system
        assert "Даже в Codex CLI не зеркаль мужской род" in request.system
        assert "я бы разложила" in request.system
        assert "я бы разложил" in request.system
        assert "Живость остаётся через конкретную позицию" in request.system
        first_message = request.messages[0]
        assert first_message["role"] == "user"
        assert "<CURRENT_MESSAGE>" in first_message["content"]
        assert "Как ты прочитал канал?" in first_message["content"]

    async def test_creates_profile(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("ok"))
        people = _mock_people()

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=people),
        ):
            await skill.execute("hello", _context(user_id=99999))

        people.get_or_create_profile.assert_called_once_with(99999)
        people.record_interaction.assert_called_once_with(99999)


class TestModeSpecificPrompts:
    async def test_assistant_intro_prompt_for_new_user(self, tmp_path: Path) -> None:
        """First contact in assistant mode uses introduction prompt."""
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        from pathlib import Path

        personality_dir = Path(ws) / "personality"
        (personality_dir / "public_identity.md").write_text(
            "Публичная идентичность Жвуши."
        )
        (personality_dir / "public_core.md").write_text(
            "Публичный голос Жвуши: живая, честная, с характером."
        )
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("Привет! Я Жвуша."))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people(interaction_count=1)),
        ):
            result = await skill.execute("привет", _context(mode="assistant"))

        assert result.success is True
        system = mock_router.generate.call_args.args[0].system
        assert "Представься коротко" in system
        assert "Режим: помощник" not in system
        assert "Публичная идентичность Жвуши." in system
        assert "Публичный голос Жвуши: живая, честная, с характером." in system

    async def test_assistant_normal_prompt_after_intro(self, tmp_path: Path) -> None:
        """After 2+ interactions, assistant mode uses normal prompt."""
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("Чем помочь?"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people(interaction_count=3)),
        ):
            result = await skill.execute("привет", _context(mode="assistant"))

        assert result.success is True
        system = mock_router.generate.call_args.args[0].system
        assert "Режим: помощник" in system
        assert "Представься коротко" not in system

    async def test_personal_mode_ignores_interaction_count(
        self, tmp_path: Path
    ) -> None:
        """Personal mode always uses personal prompt regardless of count."""
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        skill = ChatResponseSkill()

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("Привет!"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people(interaction_count=1)),
        ):
            await skill.execute("привет", _context(mode="personal"))

        system = mock_router.generate.call_args.args[0].system
        assert "Никогда не будь формальной" in system
        assert "Представься коротко" not in system
        assert "I am Zhvusha." in system
        assert "Непереписываемая личность" in system


class TestRetrieval:
    async def test_personal_mode_uses_retrieval(self, tmp_path: Path) -> None:
        """Personal mode with decision_engine calls retrieve_for_question."""
        from src.core.decision import RetrievalResult

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        mock_engine = AsyncMock()
        mock_engine.retrieve_for_question = AsyncMock(
            return_value=RetrievalResult(
                memory_context="вчера обсуждали aiogram",
                file_context="### diary/2026-04-02.md\nхороший день",
                depth="MEMORY",
            )
        )

        skill = ChatResponseSkill(decision_engine=mock_engine)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(
            return_value=_llm_resp("Отвечаю с контекстом!")
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("как прошло утро?", _context(mode="personal"))

        assert result.success is True
        mock_engine.retrieve_for_question.assert_awaited_once()
        call_args = mock_engine.retrieve_for_question.call_args
        assert call_args[0][0] == "как прошло утро?"
        llm_prompt = mock_router.generate.call_args.args[0].prompt
        assert "aiogram" in llm_prompt
        assert "хороший день" in llm_prompt

    async def test_assistant_mode_skips_retrieval(self, tmp_path: Path) -> None:
        """Non-personal mode should not call decision_engine."""
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        mock_engine = AsyncMock()
        skill = ChatResponseSkill(decision_engine=mock_engine)

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("ok"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people(interaction_count=3)),
        ):
            await skill.execute("hello", _context(mode="assistant"))

        mock_engine.retrieve_for_question.assert_not_awaited()

    async def test_quick_response_skips_second_llm_call(self, tmp_path: Path) -> None:
        """QUICK with quick_response uses it directly, no second LLM call."""
        from src.core.decision import RetrievalResult

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        mock_engine = AsyncMock()
        mock_engine.retrieve_for_question = AsyncMock(
            return_value=RetrievalResult(
                depth="QUICK", quick_response="привет, Никита!"
            )
        )

        skill = ChatResponseSkill(decision_engine=mock_engine)

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("should not be called"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("привет", _context(mode="personal"))

        assert result.response == "привет, Никита!"
        mock_router.generate.assert_not_awaited()

    async def test_transport_probe_skips_context_planner(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        mock_engine = AsyncMock()
        mock_engine.retrieve_for_question = AsyncMock()
        skill = ChatResponseSkill(decision_engine=mock_engine)

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("pong"))

        context = AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            message_id=1,
            bot=None,
            metadata={
                "source": "vscode",
                "interface": "vscode",
                "source_actor": "codex",
                "chat_log_id": "vscode",
                "skip_response_log": True,
            },
        )

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("ответь коротко pong", context)

        assert result.response == "pong"
        mock_engine.retrieve_for_question.assert_not_awaited()
        mock_router.generate_with_tools.assert_not_awaited()
        assert mock_router.generate.call_args.args[0].tier == "worker"

    async def test_context_planner_timeout_falls_back_to_chat(
        self,
        tmp_path: Path,
    ) -> None:
        class SlowContextPlanner:
            def __init__(self) -> None:
                self.calls = 0

            async def retrieve_for_question(
                self, *_args: object, **_kwargs: object
            ) -> object:
                self.calls += 1
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)
        settings.chat_decision_context_timeout_seconds = 0.01

        planner = SlowContextPlanner()
        skill = ChatResponseSkill(decision_engine=planner)  # type: ignore[arg-type]

        mock_router = AsyncMock()
        mock_router.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
        mock_router.generate = AsyncMock(return_value=_llm_resp("обычный ответ"))

        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=mock_router),
            patch(_PATCH_PEOPLE, return_value=_mock_people()),
        ):
            result = await skill.execute("как дела?", _context(mode="personal"))

        assert result.response == "обычный ответ"
        assert planner.calls == 1
        mock_router.generate.assert_awaited_once()


class TestDependencyInjection:
    async def test_accepts_llm_router_and_log_callback(self, tmp_path: Path) -> None:
        """ChatResponseSkill accepts llm_router and log_bot_response via DI."""
        ws = str(tmp_path / "ws")
        _setup_workspace(ws)
        settings = _settings(ws)

        mock_router = AsyncMock()
        mock_router.generate = AsyncMock(return_value=_llm_resp("ok"))
        log_calls: list[dict[str, Any]] = []

        def _capture_log(**kwargs: Any) -> None:
            log_calls.append(kwargs)

        skill = ChatResponseSkill(
            llm_router=mock_router,
            log_bot_response_callback=_capture_log,
        )

        # Use assistant mode to avoid the agentic loop path (personal mode
        # calls generate_with_tools first, falling back to generate only on
        # NotImplementedError).
        with (
            patch(_PATCH_SETTINGS, return_value=settings),
            patch(_PATCH_ROUTER, return_value=AsyncMock()),
            patch(_PATCH_PEOPLE, return_value=_mock_people(interaction_count=3)),
        ):
            result = await skill.execute("привет", _context(mode="assistant"))

        assert result.success is True
        assert len(log_calls) == 1
        assert log_calls[0]["text"] == "ok"
