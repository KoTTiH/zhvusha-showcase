"""Tests for ``chat_self_coding.skill`` (Phase 40).

The skill is the chat-mode orchestrator. It owns no business logic of
its own — it routes user messages to the existing ``ideation_to_spec``,
``spec_command`` and ``implement_spec`` skills based on classified
intent, while keeping the user in a friendlier conversational shell.

We verify the routing contract: each intent calls the right downstream,
state transitions are correct, the entry / exit commands are wired,
and HTML responses use the block formatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText
from src.skills.base import AgentContext, SkillResult

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeStateStore:
    """In-memory ``StateStore`` for tests."""

    def __init__(self) -> None:
        self._kv: dict[int, Any] = {}

    async def load(self, user_id: int) -> Any | None:
        return self._kv.get(user_id)

    async def save(self, state: Any) -> None:
        self._kv[state.user_id] = state

    async def clear(self, user_id: int) -> None:
        self._kv.pop(user_id, None)


class BrokenStateStore:
    """State store double that simulates a Redis outage."""

    async def load(self, user_id: int) -> Any | None:
        del user_id
        raise ConnectionError("redis down")

    async def save(self, state: Any) -> None:
        del state
        raise AssertionError("save must not run")

    async def clear(self, user_id: int) -> None:
        del user_id
        raise AssertionError("clear must not run")


def _ctx(
    user_id: int = 1,
    *,
    bot: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentContext:
    return AgentContext(
        user_id=user_id,
        chat_id=user_id,
        mode="personal",
        bot=bot,
        metadata=metadata or {},
    )


def _engineering_dialogue_metadata() -> dict[str, Any]:
    return {
        "dialogue_state": {
            "chat_id": "1",
            "selected_skill": "chat_response",
            "pending_action": "",
            "last_user_message": (
                "сделай live browser link в телегу с persistent session"
            ),
            "last_assistant_response": (
                "делаю. беру за основу live browser link в телегу + "
                "persistent session. следующим шагом нужен полный рабочий "
                "контекст/репа, чтобы реально врезать это в runtime."
            ),
        },
        "recent_decision_messages": (
            "Собеседник: сделай live browser link в телегу с persistent session\n"
            "Жвуша: делаю. беру за основу live browser link в телегу + "
            "persistent session. следующим шагом нужен полный рабочий "
            "контекст/репа, чтобы реально врезать это в runtime."
        ),
    }


def _fake_skill_returning(response: str = "") -> AsyncMock:
    skill = AsyncMock()
    skill.execute = AsyncMock(return_value=SkillResult(success=True, response=response))
    return skill


@dataclass
class _Wiring:
    skill: Any
    state_store: FakeStateStore
    classifier: AsyncMock
    ideation: AsyncMock
    implement: AsyncMock
    spec: AsyncMock
    merge_handler: AsyncMock
    discussion: AsyncMock | None
    explorer: AsyncMock | None
    implementation_runner: AsyncMock | None
    task_transcript_store: Any | None


def _build(
    *,
    classifier_intent: str = "other",
    admin_id: int = 1,
    discussion_skill: AsyncMock | None = None,
    explorer_runner: AsyncMock | None = None,
    implementation_runner: AsyncMock | None = None,
    session_archive_dir: Path | None = None,
    task_transcript_store: Any | None = None,
    spec_tier_resolver: Any | None = None,
) -> _Wiring:
    """Helper: assemble the skill with stub dependencies."""
    from src.skills.chat_self_coding.intent_classifier import (
        Intent,
        IntentClassification,
    )
    from src.skills.chat_self_coding.skill import ChatSelfCodingSkill

    state_store = FakeStateStore()
    classifier = AsyncMock(
        return_value=IntentClassification(
            intent=Intent(classifier_intent), slug=None, confidence=0.9
        )
    )
    ideation = _fake_skill_returning("(technical) spec drafted")
    implement = _fake_skill_returning("(technical) editor done")
    spec = _fake_skill_returning("(technical) spec approved")
    merge_handler = AsyncMock(return_value=SkillResult(success=True, response="Слила."))

    skill = ChatSelfCodingSkill(
        admin_user_id=admin_id,
        state_store=state_store,  # type: ignore[arg-type]
        intent_classifier=classifier,
        ideation_skill=ideation,
        implement_skill=implement,
        spec_skill=spec,
        merge_handler=merge_handler,
        discussion_skill=discussion_skill,
        explorer_runner=explorer_runner,
        implementation_runner=implementation_runner,
        session_archive_dir=session_archive_dir,
        task_transcript_store=task_transcript_store,
        spec_tier_resolver=spec_tier_resolver,
    )
    return _Wiring(
        skill=skill,
        state_store=state_store,
        classifier=classifier,
        ideation=ideation,
        implement=implement,
        spec=spec,
        merge_handler=merge_handler,
        discussion=discussion_skill,
        explorer=explorer_runner,
        implementation_runner=implementation_runner,
        task_transcript_store=task_transcript_store,
    )


# ---------------------------------------------------------------------------
# Skill metadata
# ---------------------------------------------------------------------------


class TestSkillMetadata:
    def test_name_and_tier(self) -> None:
        from src.skills.chat_self_coding.skill import ChatSelfCodingSkill

        assert ChatSelfCodingSkill.name == "chat_self_coding"
        assert ChatSelfCodingSkill.llm_tier == "worker"
        assert "/код" in ChatSelfCodingSkill.triggers
        assert "/code" in ChatSelfCodingSkill.triggers
        assert "/самокодинг" in ChatSelfCodingSkill.triggers
        assert "/self_coding" in ChatSelfCodingSkill.triggers


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    async def test_entry_command_returns_high_score(self) -> None:
        w = _build()
        score = await w.skill.can_handle("/код", _ctx())
        assert score >= 0.95

    async def test_latin_entry_alias_returns_high_score(self) -> None:
        w = _build()
        score = await w.skill.can_handle("/code", _ctx())
        assert score >= 0.95

    async def test_legacy_entry_alias_returns_high_score(self) -> None:
        w = _build()
        score = await w.skill.can_handle("/самокодинг", _ctx())
        assert score >= 0.95

    async def test_non_admin_returns_zero(self) -> None:
        w = _build(admin_id=1)
        score = await w.skill.can_handle("/код", _ctx(user_id=999))
        assert score == 0.0

    async def test_text_inside_mode_returns_high_score(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        score = await w.skill.can_handle("хочу новый пресет", _ctx())
        assert score > 1.0

    async def test_slash_commands_inside_mode_are_intercepted_before_publishers(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        assert await w.skill.can_handle("/post hello", _ctx()) > 1.0
        assert await w.skill.can_handle("/post_draft publish x", _ctx()) > 1.0

    async def test_closed_session_does_not_intercept_normal_text(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(
            ChatSelfCodingState(user_id=1, stage=Stage.IDLE, is_open=False)
        )

        assert await w.skill.can_handle("обычный чат", _ctx()) == 0.0
        assert await w.skill.can_handle("/код", _ctx()) >= 0.95
        assert await w.skill.can_handle("/code", _ctx()) >= 0.95
        assert await w.skill.can_handle("/готово", _ctx()) >= 0.9

    async def test_text_outside_mode_returns_zero(self) -> None:
        w = _build()
        score = await w.skill.can_handle("просто текст", _ctx())
        assert score == 0.0

    async def test_short_confirmation_after_engineering_dialogue_enters_code_mode(
        self,
    ) -> None:
        w = _build()

        score = await w.skill.can_handle(
            "Делай",
            _ctx(metadata=_engineering_dialogue_metadata()),
        )

        assert score > 1.0

    async def test_short_confirmation_does_not_hijack_other_pending_action(
        self,
    ) -> None:
        w = _build()
        metadata = _engineering_dialogue_metadata()
        metadata["dialogue_state"]["pending_action"] = "telegram_send"

        score = await w.skill.can_handle("Делай", _ctx(metadata=metadata))

        assert score == 0.0

    async def test_state_store_outage_does_not_intercept_normal_chat(self) -> None:
        w = _build()
        w.skill._state_store = BrokenStateStore()

        assert await w.skill.can_handle("обычный чат", _ctx()) == 0.0
        assert await w.skill.can_handle("/готово", _ctx()) == 0.0
        assert await w.skill.can_handle("/код", _ctx()) >= 0.95

    async def test_assistant_mode_returns_zero(self) -> None:
        """Chat-mode is admin/personal-only; assistant requests pass through."""
        w = _build()
        ctx = AgentContext(user_id=1, chat_id=1, mode="assistant")
        score = await w.skill.can_handle("/код", ctx)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Entry / exit
# ---------------------------------------------------------------------------


class TestEntryAndExit:
    async def test_entry_creates_idle_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage

        w = _build()
        await w.skill.execute("/код", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE
        assert state.is_open is True
        assert state.code_task_id.startswith("code-task-")

    async def test_entry_reopens_existing_session_without_losing_context(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="kept-spec",
                is_open=False,
                recent_messages=("Никита: обсудили host ops",),
                code_task_id="code-task-existing",
            )
        )

        result = await w.skill.execute("/код", _ctx())

        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.active_spec_slug == "kept-spec"
        assert state.is_open is True
        assert state.recent_messages == ("Никита: обсудили host ops",)
        assert state.code_task_id == "code-task-existing"
        assert "kept-spec" in result.response

    async def test_clear_resets_context_but_keeps_code_mode_open(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="old-spec",
                recent_messages=("Никита: старый контекст",),
                recovery_kind="run_spec",
                recovery_text="old-spec",
                recovery_error="ошибка",
            )
        )

        result = await w.skill.execute("/clear", _ctx())

        assert result.success is True
        state = await w.state_store.load(1)
        assert state is not None
        assert state.is_open is True
        assert state.stage == Stage.IDLE
        assert state.active_spec_slug is None
        assert state.recent_messages == ()
        assert state.recovery_kind is None
        assert state.active_goal is None
        assert state.readonly_codex_session_id is None

    async def test_goal_command_sets_codex_room_goal(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute(
            "goal довести /код до уровня терминального Codex", _ctx()
        )

        assert result.success is True
        state = await w.state_store.load(1)
        assert state is not None
        assert state.active_goal == "довести /код до уровня терминального Codex"
        assert any("зафиксировал цель" in item for item in state.recent_messages)

    async def test_compact_resets_readonly_codex_thread_but_preserves_summary(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                active_goal="убрать ephemeral Codex",
                recent_messages=("Никита: дорого", "Жвуша: причина в ephemeral"),
                readonly_codex_session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
                recovery_kind="run_spec",
                recovery_text="x",
                recovery_error="old blocker",
            )
        )

        result = await w.skill.execute("compact", _ctx())

        assert result.success is True
        state = await w.state_store.load(1)
        assert state is not None
        assert state.readonly_codex_session_id is None
        assert state.recovery_kind is None
        assert state.compact_summary is not None
        assert "убрать ephemeral Codex" in state.compact_summary
        assert state.recent_messages == (
            f"Сжатый контекст /код: {state.compact_summary}",
        )

    async def test_latin_entry_alias_creates_idle_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage

        w = _build()
        await w.skill.execute("/code", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE

    async def test_legacy_latin_entry_alias_creates_idle_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage

        w = _build()
        await w.skill.execute("/self_coding", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE

    async def test_entry_responds_with_welcome_in_html(self) -> None:
        """When no bot is wired (unit-test default), the HTML welcome
        body comes back via SkillResult.response — back-compat with the
        original Phase 40 contract."""
        w = _build()
        result = await w.skill.execute("/код", _ctx())
        assert "<b>" in result.response
        assert "/код" in result.response.lower()

    async def test_entry_sends_welcome_via_bot_when_available(self) -> None:
        """When a bot is on context, the skill must send the welcome via
        ``bot.send_message(parse_mode="HTML")`` directly — bypassing the
        dispatcher's markdown-to-HTML conversion which would escape the
        already-HTML tags into literal <b>."""
        bot = AsyncMock()
        bot.send_message = AsyncMock()
        ctx = AgentContext(user_id=1, chat_id=1, mode="personal", bot=bot)

        w = _build()
        result = await w.skill.execute("/код", ctx)

        bot.send_message.assert_awaited_once()
        call_kwargs = bot.send_message.call_args.kwargs
        assert call_kwargs.get("parse_mode") == "HTML"
        assert "<b>" in call_kwargs["text"]
        # Response must be empty so the dispatcher doesn't double-send.
        assert result.response == ""

    async def test_entry_returns_welcome_for_vscode_context_with_bot(self) -> None:
        """VS Code bridge needs a returned response even when bot is wired.

        The bridge logs ``SkillResult.response`` into the visible VS Code chat;
        sending only through Telegram bot makes /код look silent to Codex.
        """
        bot = AsyncMock()
        bot.send_message = AsyncMock()
        ctx = AgentContext(
            user_id=1,
            chat_id=1,
            mode="personal",
            bot=bot,
            metadata={
                "source": "vscode",
                "interface": "vscode",
                "return_response_text": True,
            },
        )

        w = _build()
        result = await w.skill.execute("/код", ctx)

        bot.send_message.assert_not_awaited()
        assert "<b>" in result.response
        assert "/код" in result.response.lower()

    async def test_exit_closes_mode_but_keeps_session(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="exit")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                recent_messages=("Никита: важный контекст",),
            )
        )
        result = await w.skill.execute("выход", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.is_open is False
        assert state.recent_messages == ("Никита: важный контекст",)
        assert "сессию оставила" in result.response.lower()

    async def test_exit_via_keyword_does_not_destroy_session(self) -> None:
        """Pure-keyword «выход» should close only the chat-mode room."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build()  # classifier set to "other" — would NOT be EXIT if called
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await w.skill.execute("выход", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.is_open is False

    async def test_complete_trigger_archives_and_clears_session(
        self, tmp_path: Path
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        archive_dir = tmp_path / "self_coding_sessions"
        w = _build(session_archive_dir=archive_dir)
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                is_open=False,
                recent_messages=("Никита: старый хвост",),
            )
        )

        result = await w.skill.execute("/готово", _ctx())

        assert await w.state_store.load(1) is None
        assert "сохранила" in result.response.lower()
        archived = list(archive_dir.glob("*.md"))
        assert len(archived) == 1
        archive_text = archived[0].read_text(encoding="utf-8")
        assert "Никита: старый хвост" in archive_text
        assert "stage: idle" in archive_text


# ---------------------------------------------------------------------------
# Routing — every intent must be handled
# ---------------------------------------------------------------------------


class TestIntentRouting:
    async def test_create_proxies_to_ideation_skill(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await w.skill.execute("хочу новый пресет для багов", _ctx())
        w.ideation.execute.assert_awaited_once()
        # The forwarded message must invoke the legacy slash command so
        # ideation_skill's existing pipeline runs unchanged.
        sent_message = w.ideation.execute.call_args.args[0]
        assert sent_message.startswith("/spec_create")
        assert "багов" in sent_message

    async def test_implicit_entry_from_chat_followup_starts_spec_creation(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import TaskPhase

        w = _build(classifier_intent="other")
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=True,
                response="",
                metadata={"slug": "live-browser-human-verification", "tier": 3},
            )
        )

        result = await w.skill.execute(
            "Делай",
            _ctx(metadata=_engineering_dialogue_metadata()),
        )

        assert result.success is True
        w.ideation.execute.assert_awaited_once()
        sent_message = w.ideation.execute.await_args.args[0]
        assert sent_message.startswith("/spec_create ")
        assert "Диалог до входа в /код" in sent_message
        assert "live browser link" in sent_message
        assert "persistent session" in sent_message
        assert "Текущая команда Никиты" in sent_message
        state = await w.state_store.load(1)
        assert state is not None
        assert state.is_open is True
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.task_phase is TaskPhase.APPROVAL
        assert state.active_spec_slug == "live-browser-human-verification"

    async def test_create_binds_slug_from_metadata_and_transitions_to_pending_approval(
        self,
    ) -> None:
        """After Architect drafts a spec it returns the slug in
        ``SkillResult.metadata``; chat-mode must record it on the
        session state so the next ``approve`` / ``reject`` knows which
        spec to act on. Stage flips DRAFTING → PENDING_APPROVAL because
        Architect runs synchronously: by the time we get back, the plan
        is ready."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase

        w = _build(classifier_intent="create_spec")
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=True,
                response="(technical reply)",
                metadata={"slug": "my-new-spec", "tier": 1},
            )
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await w.skill.execute("хочу новый пресет", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.active_spec_slug == "my-new-spec"
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.task_phase is TaskPhase.APPROVAL

    async def test_create_sends_progress_message_before_architect_runs(
        self,
    ) -> None:
        """The progress message must arrive BEFORE Architect blocks."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")

        timeline: list[str] = []
        bot = AsyncMock()

        async def record_send(*_args: Any, **_kwargs: Any) -> None:
            timeline.append("progress_message_sent")

        async def record_ideation(*_args: Any, **_kwargs: Any) -> SkillResult:
            timeline.append("architect_executed")
            state = await w.state_store.load(1)
            assert state is not None
            assert state.stage is Stage.DRAFTING
            assert state.drafting_started_at_epoch is not None
            return SkillResult(
                success=True, response="", metadata={"slug": "x", "tier": 1}
            )

        bot.send_message = record_send
        w.ideation.execute = record_ideation

        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        ctx = AgentContext(user_id=1, chat_id=1, mode="personal", bot=bot)
        await w.skill.execute("хочу пресет", ctx)

        assert timeline == ["progress_message_sent", "architect_executed"]

    async def test_create_progress_message_uses_confirmed_stage_and_identity(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        bot = AsyncMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        bot.edit_message_text = AsyncMock()
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=True, response="", metadata={"slug": "x", "tier": 1}
            )
        )

        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        ctx = AgentContext(user_id=1, chat_id=1, mode="personal", bot=bot)
        await w.skill.execute("хочу пресет", ctx)

        sent_text = bot.send_message.await_args.kwargs["text"]
        assert "Жвуша" in sent_text
        assert "10%" not in sent_text
        assert "[##------------------]" not in sent_text
        assert "Этап: приём задачи" in sent_text
        assert "Подтверждённый этап:" in sent_text
        bot.edit_message_text.assert_awaited()
        final_text = bot.edit_message_text.await_args.kwargs["text"]
        assert "100%" not in final_text
        assert "План собран" in final_text

    async def test_create_progress_message_finishes_on_architect_failure(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        bot = AsyncMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        bot.edit_message_text = AsyncMock()
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=False,
                response="Spec validation failed.",
            )
        )

        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        ctx = AgentContext(user_id=1, chat_id=1, mode="personal", bot=bot)
        await w.skill.execute("хочу пресет", ctx)

        final_text = bot.edit_message_text.await_args.kwargs["text"]
        assert "100%" not in final_text
        assert "не собрался" in final_text

    async def test_create_returns_to_idle_when_architect_fails(self) -> None:
        """If Architect couldn't draft a spec, drop back to IDLE so the
        user can retry without a stuck PENDING_APPROVAL state."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=False, response="Architect SDK не ответил."
            )
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await w.skill.execute("хочу пресет", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE
        assert state.active_spec_slug is None
        assert state.recovery_kind == "create_spec"
        assert state.recovery_text is not None
        assert "хочу пресет" in state.recovery_text

    async def test_create_surfaces_architect_failure_reason(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=False,
                response=(
                    "Spec validation failed:\n"
                    "blast_radius.3: Input should be a valid string"
                ),
            )
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("Контекст:\nнадо исправить приветствие", _ctx())

        assert result.success is False
        assert "blast_radius" in result.response
        assert "valid string" in result.response
        assert "обсудим" in result.response

    async def test_create_failure_enters_discussion_and_resume_retries_spec(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassification,
            Stage,
        )
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Поняла: оставляем полный scope.")
        w = _build(
            classifier_intent="create_spec",
            discussion_skill=discussion,
        )
        w.classifier.side_effect = [
            IntentClassification(intent=Intent.CREATE_SPEC),
            IntentClassification(intent=Intent.OTHER),
        ]
        w.ideation.execute = AsyncMock(
            side_effect=[
                SkillResult(success=False, response="Architect вернул не YAML."),
                SkillResult(
                    success=True,
                    response="",
                    metadata={"slug": "visual-pipeline", "tier": 3},
                ),
            ]
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        first = await w.skill.execute("оформи план", _ctx())
        state_after_fail = await w.state_store.load(1)

        assert first.success is False
        assert state_after_fail is not None
        assert state_after_fail.stage == Stage.IDLE
        assert state_after_fail.recovery_kind == "create_spec"

        await w.skill.execute(
            "Да, делаем полный вариант, без маленького slice.", _ctx()
        )
        discussion.execute.assert_awaited_once()
        assert w.ideation.execute.await_count == 1

        second = await w.skill.execute("продолжай", _ctx())
        resumed_prompt = w.ideation.execute.await_args.args[0]
        state_after_resume = await w.state_store.load(1)

        assert second.success is True
        assert "Повторная попытка составить spec после ошибки" in resumed_prompt
        assert "оставляем полный scope" in resumed_prompt
        assert state_after_resume is not None
        assert state_after_resume.stage == Stage.PENDING_APPROVAL
        assert state_after_resume.active_spec_slug == "visual-pipeline"
        assert state_after_resume.recovery_kind is None

    async def test_long_recovery_resume_phrase_retries_without_discussion_llm(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("не должно вызываться")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        w.implement.execute = AsyncMock(
            return_value=SkillResult(success=True, response="done")
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="channel-visual-pipeline-v2",
                recovery_kind="run_spec",
                recovery_text="channel-visual-pipeline-v2",
                recovery_error="tier был 2",
            )
        )

        result = await w.skill.execute(
            "Я поправил, теперь максимальный тир 3, продолжай разработку",
            _ctx(),
        )

        assert result.success is True
        w.spec.execute.assert_awaited_once()
        forwarded = w.spec.execute.await_args.args[0]
        assert "approve" in forwarded
        assert "channel-visual-pipeline-v2" in forwarded
        w.implement.execute.assert_awaited_once()
        discussion.execute.assert_not_awaited()

    async def test_create_surfaces_architect_clarification_as_dialogue(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=False,
                response="Сохранять старый fallback или можно удалить?",
                metadata={"needs_clarification": True},
            )
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("оформи план", _ctx())

        assert result.success is True
        assert "fallback" in result.response
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE
        assert state.recovery_kind == "create_spec"
        assert any("fallback" in m for m in state.recent_messages)

    async def test_create_host_ops_activation_stops_before_architect(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                recent_messages=(
                    "Никита: /restart код уже есть",
                    "Жвуша: нужно включить systemd supervisor",
                ),
            )
        )

        result = await w.skill.execute(
            "Включи runtime-контур: BOT_RESTART_ENABLED=true, "
            "systemd daemon-reload и enable --now, чтобы всё работало.",
            _ctx(),
        )

        assert result.success is True
        assert "host/runtime" in result.response
        assert "systemd" in result.response
        w.ideation.execute.assert_not_awaited()
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE
        assert any("host/runtime" in m for m in state.recent_messages)

    async def test_approve_proxies_to_spec_skill_with_active_slug(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        await w.skill.execute("делай", _ctx())
        w.spec.execute.assert_awaited_once()
        forwarded = w.spec.execute.call_args.args[0]
        assert "approve" in forwarded
        assert "my-spec" in forwarded

    async def test_tier3_approval_marks_classifier_context_as_ai_required(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(
            classifier_intent="other",
            spec_tier_resolver=lambda slug: 3 if slug == "tier3-spec" else 1,
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="tier3-spec",
            )
        )

        await w.skill.execute("делай", _ctx())

        classifier_context = w.classifier.await_args.args[0]
        assert classifier_context.requires_ai_approval is True
        w.spec.execute.assert_not_awaited()

    async def test_approve_auto_runs_editor_after_approval(self) -> None:
        """Phase 40 plan: «делай» в pending_approval → 🔧 Подготовка
        (Editor cycle starts) without an extra «запускай» step. The
        approve handler must auto-run ``implement_skill`` once the spec
        is approved."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        await w.skill.execute("делай", _ctx())

        # spec_command (approve) AND implement_spec (run) both invoked.
        w.spec.execute.assert_awaited_once()
        w.implement.execute.assert_awaited_once()
        run_message = w.implement.execute.call_args.args[0]
        assert "spec_run" in run_message
        assert "my-spec" in run_message

    async def test_approve_can_run_implementation_through_agent_runtime(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        implementation_runner = AsyncMock(
            return_value=SkillResult(
                success=True,
                response="agent runtime done",
                metadata={"agent_job_id": "job-1"},
            )
        )
        w = _build(
            classifier_intent="approve",
            implementation_runner=implementation_runner,
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
                recent_messages=("Никита: оформи план", "Жвуша: план готов"),
            )
        )

        await w.skill.execute("делай", _ctx())

        w.spec.execute.assert_awaited_once()
        w.implement.execute.assert_not_awaited()
        implementation_runner.assert_awaited_once()
        call = implementation_runner.await_args.kwargs
        assert call["slug"] == "my-spec"
        assert call["recent_messages"] == (
            "Никита: оформи план",
            "Жвуша: план готов",
            "Никита: делай",
        )

    async def test_entry_tail_run_slug_starts_implementation_not_discussion(
        self,
    ) -> None:
        implementation_runner = AsyncMock(
            return_value=SkillResult(success=True, response="agent runtime done")
        )
        explorer = AsyncMock(return_value="не должен запускаться")
        w = _build(
            classifier_intent="other",
            explorer_runner=explorer,
            implementation_runner=implementation_runner,
        )

        result = await w.skill.execute(
            "/код делай fix-code-chat-implementation-profile-gate. коротко",
            _ctx(),
        )

        assert result.success is True
        w.spec.execute.assert_awaited_once()
        assert (
            "approve fix-code-chat-implementation-profile-gate"
            in (w.spec.execute.await_args.args[0])
        )
        implementation_runner.assert_awaited_once()
        assert (
            implementation_runner.await_args.kwargs["slug"]
            == "fix-code-chat-implementation-profile-gate"
        )
        explorer.assert_not_awaited()
        w.classifier.assert_not_awaited()

    async def test_long_run_slug_inside_open_room_bypasses_explorer(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        implementation_runner = AsyncMock(
            return_value=SkillResult(success=True, response="agent runtime done")
        )
        explorer = AsyncMock(return_value="read-only")
        w = _build(
            classifier_intent="other",
            explorer_runner=explorer,
            implementation_runner=implementation_runner,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        await w.skill.execute(
            "делай fix-code-chat-implementation-profile-gate. нужен write path",
            _ctx(),
        )

        implementation_runner.assert_awaited_once()
        assert (
            implementation_runner.await_args.kwargs["slug"]
            == "fix-code-chat-implementation-profile-gate"
        )
        explorer.assert_not_awaited()
        w.classifier.assert_not_awaited()

    async def test_approve_starts_agent_runtime_implementation_in_background(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase

        class BackgroundRunner:
            def __init__(self) -> None:
                self.calls = 0
                self.completion: Any | None = None

            async def __call__(self, **kwargs: Any) -> SkillResult:
                del kwargs
                raise AssertionError("sync implementation should not run")

            async def start_background(self, **kwargs: Any) -> Any:
                self.calls += 1
                self.completion = kwargs["completion_callback"]
                return SimpleNamespace(id="job-1")

        runner = BackgroundRunner()
        w = _build(
            classifier_intent="approve",
            implementation_runner=runner,  # type: ignore[arg-type]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        bot = SimpleNamespace(send_message=AsyncMock())

        result = await w.skill.execute("делай", _ctx(bot=bot))

        assert result.success is True
        assert result.response == ""
        assert runner.calls == 1
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.RUNNING
        assert state.task_phase is TaskPhase.IMPLEMENTATION

        assert runner.completion is not None
        await runner.completion(
            SkillResult(
                success=True,
                response="готово",
                metadata={"agent_job_id": "job-1"},
            )
        )
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.DONE
        assert state.task_phase is TaskPhase.DONE

    async def test_background_auto_retryable_failure_is_not_restarted(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        class BackgroundRunner:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **kwargs: Any) -> SkillResult:
                del kwargs
                raise AssertionError("sync implementation should not run")

            async def start_background(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return SimpleNamespace(id=f"job-{len(self.calls)}")

        runner = BackgroundRunner()
        w = _build(
            classifier_intent="approve",
            implementation_runner=runner,  # type: ignore[arg-type]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
                recent_messages=("Никита: media-first unsafe",),
            )
        )
        bot = SimpleNamespace(send_message=AsyncMock())

        result = await w.skill.execute("делай", _ctx(bot=bot))

        assert result.success is True
        assert len(runner.calls) == 1
        first_context = runner.calls[0]["context"]
        assert first_context.metadata["chat_self_coding_goal_attempt"] == 0

        await runner.calls[0]["completion_callback"](
            SkillResult(
                success=False,
                response="Reviewer blocked media-first publish.",
                metadata={
                    "needs_user_decision": "false",
                    "auto_retryable": "true",
                    "failure_gate": "Reviewer verdict `reject`",
                },
            )
        )

        assert len(runner.calls) == 1
        assert w.spec.execute.await_count == 1
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.recovery_kind == "run_spec"
        assert state.recovery_needs_user_decision is False

    async def test_approve_transitions_state_to_done_after_successful_editor_run(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        await w.skill.execute("делай", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.DONE

    async def test_approve_skips_run_when_spec_skill_fails(self) -> None:
        """If approve itself fails, don't try to run a spec that didn't
        get marked approved on disk."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        w.spec.execute = AsyncMock(
            return_value=SkillResult(success=False, response="approve gate failed")
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        result = await w.skill.execute("делай", _ctx())
        w.implement.execute.assert_not_awaited()
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.recovery_kind == "approve_spec"
        assert "обсудим" in result.response

    async def test_approve_returns_to_pending_approval_when_editor_dry_runs(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        w.implement.execute = AsyncMock(
            return_value=SkillResult(success=True, response="dry-run: would edit files")
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        result = await w.skill.execute("делай", _ctx())

        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.recovery_kind == "run_spec"
        assert "dry-run" in result.response

    async def test_editor_failure_enters_discussion_and_resume_retries_run(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import (
            Intent,
            IntentClassification,
            Stage,
        )
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Поняла, оставляем whitelist шире.")
        w = _build(classifier_intent="approve", discussion_skill=discussion)
        w.classifier.side_effect = [
            IntentClassification(intent=Intent.APPROVE),
            IntentClassification(intent=Intent.OTHER),
        ]
        w.implement.execute = AsyncMock(
            side_effect=[
                SkillResult(success=False, response="Commit gate failed."),
                SkillResult(success=True, response="done"),
            ]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        first = await w.skill.execute("делай", _ctx())
        state_after_fail = await w.state_store.load(1)

        assert first.success is False
        assert "Commit gate failed" in first.response
        assert state_after_fail is not None
        assert state_after_fail.stage == Stage.PENDING_APPROVAL
        assert state_after_fail.recovery_kind == "run_spec"

        await w.skill.execute("Да, поправь через тот же spec.", _ctx())
        assert w.implement.execute.await_count == 1
        discussion.execute.assert_awaited_once()

        second = await w.skill.execute("продолжай", _ctx())
        state_after_resume = await w.state_store.load(1)

        assert second.success is True
        assert w.spec.execute.await_count == 2
        forwarded = w.spec.execute.await_args.args[0]
        assert "approve" in forwarded
        assert "my-spec" in forwarded
        assert w.implement.execute.await_count == 2
        assert state_after_resume is not None
        assert state_after_resume.stage == Stage.DONE
        assert state_after_resume.recovery_kind is None

    async def test_user_decision_failure_asks_specific_question_and_blocks_auto_retry(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        w.implement.execute = AsyncMock(
            return_value=SkillResult(
                success=False,
                response="Reviewer rejected visual provenance.",
                metadata={
                    "needs_user_decision": "true",
                    "auto_retryable": "false",
                    "failure_category": "needs_user_decision",
                    "decision_question": (
                        "какие источники визуалов считаем допустимыми?"
                    ),
                },
            )
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        result = await w.skill.execute("делай", _ctx())
        state = await w.state_store.load(1)

        assert not result.success
        assert w.implement.execute.await_count == 1
        assert "какие источники визуалов" in result.response
        assert "следующий запуск заблокирован" in result.response.lower()
        assert "продолжай" not in result.response.lower()
        assert state is not None
        assert state.recovery_kind == "run_spec"
        assert state.recovery_needs_user_decision is True

    async def test_auto_retryable_editor_failure_stops_without_hidden_retry(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("не должно вызываться")
        w = _build(classifier_intent="approve", discussion_skill=discussion)
        w.implement.execute = AsyncMock(
            side_effect=[
                SkillResult(
                    success=False,
                    response="Reviewer blocked text/media ordering.",
                    metadata={
                        "needs_user_decision": "false",
                        "auto_retryable": "true",
                    },
                ),
            ]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        result = await w.skill.execute("делай", _ctx())
        state = await w.state_store.load(1)

        assert result.success is False
        assert w.implement.execute.await_count == 1
        assert w.spec.execute.await_count == 1
        discussion.execute.assert_not_awaited()
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.recovery_kind == "run_spec"
        assert state.recovery_needs_user_decision is False

    async def test_non_decision_recovery_message_is_context_not_discussion(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("не должно вызываться")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            ).with_recovery(
                kind="run_spec",
                text="my-spec",
                error="Reviewer blocked text/media ordering.",
                needs_user_decision=False,
            )
        )

        result = await w.skill.execute("ещё учти safety gate", _ctx())
        state = await w.state_store.load(1)

        assert result.success is True
        assert "технический blocker" in result.response
        assert "продолжай" in result.response
        discussion.execute.assert_not_awaited()
        assert state is not None
        assert any("ещё учти safety gate" in item for item in state.recent_messages)

    async def test_recovery_resume_passes_discussion_to_direct_implementation(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="other")
        w.implement.execute = AsyncMock(
            return_value=SkillResult(success=True, response="done")
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
                recent_messages=(
                    "Никита: reviewer прав про media-first",
                    "Никита: надо сначала публиковать текст",
                ),
            ).with_recovery(
                kind="run_spec",
                text="my-spec",
                error="Reviewer blocked media-first publish.",
                needs_user_decision=True,
                question="оставляем text-first?",
            )
        )

        result = await w.skill.execute("продолжай", _ctx())

        assert result.success is True
        implementation_context = w.implement.execute.await_args.args[1]
        recent = implementation_context.metadata["chat_self_coding_recent_messages"]
        assert "Никита: надо сначала публиковать текст" in recent
        assert "Никита: продолжай" in recent

    async def test_editor_failure_resume_metadata_is_reused_on_next_run(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        w.implement.execute = AsyncMock(
            side_effect=[
                SkillResult(
                    success=False,
                    response="Reviewer rejected.",
                    metadata={
                        "needs_user_decision": "false",
                        "failure_category": "technical_blocker",
                        "editor_codex_session_id": "codex-editor-thread-1",
                        "failed_worktree_path": "/repo-worktrees/failed",
                        "failed_worktree_label": "isolated:spec:1:1",
                        "failed_worktree_base_branch": "main",
                        "failed_worktree_base_sha": "abc123",
                    },
                ),
                SkillResult(success=True, response="done"),
            ]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        first = await w.skill.execute("делай", _ctx())
        state_after_fail = await w.state_store.load(1)

        assert not first.success
        assert state_after_fail is not None
        assert state_after_fail.editor_codex_session_id == "codex-editor-thread-1"
        assert state_after_fail.failed_worktree_path == "/repo-worktrees/failed"

        second = await w.skill.execute("продолжай", _ctx())

        assert second.success is True
        implementation_context = w.implement.execute.await_args.args[1]
        assert (
            implementation_context.metadata["chat_self_coding_editor_codex_session_id"]
            == "codex-editor-thread-1"
        )
        assert (
            implementation_context.metadata["chat_self_coding_failed_worktree_path"]
            == "/repo-worktrees/failed"
        )

    async def test_implementation_runner_receives_stable_code_task_context(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        seen_contexts: list[AgentContext] = []

        async def implementation_runner(
            *,
            slug: str,
            context: AgentContext,
            recent_messages: tuple[str, ...] = (),
        ) -> SkillResult:
            assert slug == "my-spec"
            assert "Никита: reviewer прав" in recent_messages
            seen_contexts.append(context)
            return SkillResult(success=True, response="done")

        w = _build(
            classifier_intent="approve",
            implementation_runner=implementation_runner,  # type: ignore[arg-type]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
                recent_messages=("Никита: reviewer прав",),
                code_task_id="code-task-fixed",
            )
        )

        result = await w.skill.execute("делай", _ctx())

        assert result.success is True
        assert seen_contexts
        assert (
            seen_contexts[0].metadata["chat_self_coding_code_task_id"]
            == "code-task-fixed"
        )
        assert seen_contexts[0].metadata["chat_self_coding_goal_attempt"] == 0

    async def test_reject_proxies_to_spec_skill(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="reject")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        await w.skill.execute("не надо", _ctx())
        w.spec.execute.assert_awaited_once()
        forwarded = w.spec.execute.call_args.args[0]
        assert "reject" in forwarded
        assert "my-spec" in forwarded

    async def test_run_proxies_to_implement_skill(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="run_spec")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )
        await w.skill.execute("запускай", _ctx())
        w.implement.execute.assert_awaited_once()
        forwarded = w.implement.execute.call_args.args[0]
        assert "spec_run" in forwarded
        assert "my-spec" in forwarded

    async def test_status_returns_current_stage(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="status")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.RUNNING,
                active_spec_slug="my-spec",
            )
        )
        result = await w.skill.execute("где мы", _ctx())
        # Status response mentions the slug and the stage in human terms.
        assert "my-spec" in result.response
        assert (
            "пиш" in result.response.lower()  # «пишу» / «пишет»
            or "работа" in result.response.lower()
        )

    async def test_other_returns_clarification(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="other")
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        result = await w.skill.execute("???", _ctx())
        # Helpful clarification — not silent.
        assert result.response
        assert "выход" in result.response.lower()

    async def test_other_in_idle_delegates_to_discussion_skill_when_available(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Давай разберём без плана пока.")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("хочу сначала обсудить идею", _ctx())

        assert result.response == "Давай разберём без плана пока."
        discussion.execute.assert_awaited_once()
        sent_message = discussion.execute.call_args.args[0]
        assert sent_message == "хочу сначала обсудить идею"
        w.ideation.execute.assert_not_awaited()
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE

    async def test_discussion_inside_code_mode_suppresses_memory_proposals(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Давай обсудим без плана пока.")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        await w.skill.execute("хочу сначала обсудить идею", _ctx())

        discussion_context = discussion.execute.await_args.args[1]
        assert discussion_context.metadata["chat_self_coding"] is True
        assert discussion_context.metadata["suppress_memory_proposals"] is True

    async def test_other_after_plan_stays_discussion_and_does_not_run(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Да, это можно ещё обсудить.")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="my-spec",
            )
        )

        result = await w.skill.execute("давай ещё подумаем", _ctx())

        assert result.response == "Да, это можно ещё обсудить."
        discussion.execute.assert_awaited_once()
        w.spec.execute.assert_not_awaited()
        w.implement.execute.assert_not_awaited()
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.active_spec_slug == "my-spec"

    async def test_discussion_response_is_kept_in_recent_context(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Я бы сначала сузила правило приветствий.")
        w = _build(classifier_intent="other", discussion_skill=discussion)
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        await w.skill.execute("что думаешь про это поведение?", _ctx())

        state = await w.state_store.load(1)
        assert state is not None
        assert any("что думаешь" in m for m in state.recent_messages)
        assert any("сначала сузила" in m for m in state.recent_messages)

    async def test_incoming_material_preamble_does_not_start_explorer(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("старый чат-ответ")
        explorer = AsyncMock(return_value="Пока самого поста в контексте нет.")
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("Щас скину тебе пост, посмотри его", _ctx())

        assert "сначала дождусь" in result.response
        explorer.assert_not_awaited()
        discussion.execute.assert_not_awaited()
        state = await w.state_store.load(1)
        assert state is not None
        assert any("сначала дождусь" in m for m in state.recent_messages)

    async def test_non_repo_post_uses_plain_discussion_with_explorer_available(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Разобрала пост без чтения репозитория.")
        explorer = AsyncMock(return_value="не должен запускаться")
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute(
            "Anthropic представил режим сновидений для агентов\n\n"
            "Dreaming анализирует прошлые сессии и память.",
            _ctx(),
        )

        assert result.response == "Разобрала пост без чтения репозитория."
        explorer.assert_not_awaited()
        discussion.execute.assert_awaited_once()

    async def test_product_discussion_with_code_words_does_not_start_explorer(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Я бы закрепила это как правило визуала.")
        explorer = AsyncMock(return_value="не должен запускаться")
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute(
            "Я хочу чтобы ты генерировала визуал через gpt image 2, "
            "а скриншоты из интернета вставляла для внешних тем, "
            "но не скриншоты твоего кода. Что думаешь?",
            _ctx(),
        )

        assert result.response == "Я бы закрепила это как правило визуала."
        explorer.assert_not_awaited()
        discussion.execute.assert_awaited_once()

    async def test_visual_channel_discussion_does_not_start_explorer(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Я бы сделала полный visual pipeline.")
        explorer = AsyncMock(return_value="не должен запускаться")
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute(
            "Я хочу чтобы ты генерировала визуал через gpt image 2, "
            "схемы и рисунки, а для внешних тем вставляла скриншоты "
            "из интернета. Посмотри ai каналы и свой канал. Что думаешь?",
            _ctx(),
        )

        assert result.response == "Я бы сделала полный visual pipeline."
        explorer.assert_not_awaited()
        discussion.execute.assert_awaited_once()

    def test_explorer_prompt_keeps_zhvusha_voice_not_capsule(self) -> None:
        from src.skills.chat_self_coding.skill import _EXPLORER_SYSTEM_PROMPT

        lower = _EXPLORER_SYSTEM_PROMPT.lower()
        assert "женском роде" in lower
        assert "summary/finding/source" in lower
        assert "посмотрела" in lower

    async def test_other_uses_read_only_explorer_when_available(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("старый чат-ответ")
        explorer = AsyncMock(return_value="Проверила код: проблема в dispatcher.")
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("изучи код и скажи что происходит", _ctx())

        assert result.response == "Проверила код: проблема в dispatcher."
        explorer.assert_awaited_once()
        discussion.execute.assert_not_awaited()
        call_kwargs = explorer.await_args.kwargs
        assert "read-only" in call_kwargs["system_prompt"].lower()
        assert "изучи код" in call_kwargs["user_prompt"]
        assert callable(call_kwargs["progress_callback"]) or (
            call_kwargs["progress_callback"] is None
        )
        state = await w.state_store.load(1)
        assert state is not None
        assert any("Проверила код" in m for m in state.recent_messages)

    async def test_explorer_reuses_and_records_persistent_codex_thread(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        async def explorer(**kwargs: Any) -> str:
            assert kwargs["session_id"] == "019e1cf5-a63c-7ca1-a44e-44e555239799"
            assert kwargs["persist_session"] is True
            await kwargs["session_callback"]("019e1cf5-a63c-7ca1-a44e-44e555239799")
            return "Проверила код в той же Codex-сессии."

        w = _build(
            classifier_intent="other",
            explorer_runner=explorer,  # type: ignore[arg-type]
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                readonly_codex_session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
            )
        )

        result = await w.skill.execute("изучи код и проверь путь", _ctx())

        assert result.response == "Проверила код в той же Codex-сессии."
        state = await w.state_store.load(1)
        assert state is not None
        assert state.readonly_codex_session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"

    async def test_explorer_prompt_contains_discussion_context_and_attachment_paths(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        explorer = AsyncMock(return_value="Вложение посмотрела.")
        w = _build(classifier_intent="other", explorer_runner=explorer)
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                recent_messages=(
                    "Никита: хочу обсудить экран",
                    "Никита прислал вложение для /код\n"
                    "- absolute_path: /workspace/uploads/screen.png",
                ),
            )
        )

        await w.skill.execute("посмотри фото и код рядом", _ctx())

        prompt = explorer.await_args.kwargs["user_prompt"]
        assert "хочу обсудить экран" in prompt
        assert "/workspace/uploads/screen.png" in prompt
        assert "посмотри фото" in prompt
        assert "не создавай spec" in prompt.lower()

    async def test_explorer_failure_falls_back_to_plain_discussion(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        discussion = _fake_skill_returning("Ок, обсудим без чтения кода.")
        explorer = AsyncMock(side_effect=RuntimeError("codex unavailable"))
        w = _build(
            classifier_intent="other",
            discussion_skill=discussion,
            explorer_runner=explorer,
        )
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))

        result = await w.skill.execute("проверь код", _ctx())

        assert result.response == "Ок, обсудим без чтения кода."
        explorer.assert_awaited_once()
        discussion.execute.assert_awaited_once()

    async def test_create_after_discussion_forwards_context_to_ideation(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="create_spec")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                recent_messages=(
                    "Никита: хочу обсудить приветствие",
                    "Жвуша: Я бы убрала театральность и оставила живость.",
                ),
            )
        )

        await w.skill.execute("оформи план", _ctx())

        w.ideation.execute.assert_awaited_once()
        forwarded = w.ideation.execute.call_args.args[0]
        assert forwarded.startswith("/spec_create")
        assert "хочу обсудить приветствие" in forwarded
        assert "убрала театральность" in forwarded
        assert "оформи план" in forwarded

    async def test_idle_run_trigger_after_discussion_does_not_start_spec(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="approve")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                recent_messages=(
                    "Никита: давай обсудим проблему приветствий",
                    "Жвуша: Можно сделать правило мягче.",
                ),
            )
        )

        result = await w.skill.execute("делай", _ctx())

        w.ideation.execute.assert_not_awaited()
        w.spec.execute.assert_not_awaited()
        w.implement.execute.assert_not_awaited()
        assert "сначала" in result.response.lower()

    async def test_create_after_pending_discussion_replaces_active_spec(
        self, tmp_path: Path
    ) -> None:
        import json

        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState
        from src.skills.chat_self_coding.task_transcript import (
            FileTaskTranscriptStore,
        )

        transcript_store = FileTaskTranscriptStore(tmp_path)
        w = _build(
            classifier_intent="create_spec",
            task_transcript_store=transcript_store,
        )
        w.ideation.execute = AsyncMock(
            return_value=SkillResult(
                success=True,
                response="(technical reply)",
                metadata={"slug": "new-spec", "tier": 2},
            )
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.PENDING_APPROVAL,
                active_spec_slug="old-spec",
                code_task_id="code-task-old",
                recent_messages=(
                    "Никита: старый план не учитывает guard",
                    "Жвуша: Тогда надо пересобрать spec.",
                ),
            )
        )

        await w.skill.execute("пересобери план", _ctx())

        w.ideation.execute.assert_awaited_once()
        forwarded = w.ideation.execute.call_args.args[0]
        ideation_context = w.ideation.execute.call_args.args[1]
        assert "старый план" in forwarded
        assert "пересобери план" in forwarded
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.PENDING_APPROVAL
        assert state.active_spec_slug == "new-spec"
        assert state.code_task_id.startswith("code-task-")
        assert state.code_task_id != "code-task-old"
        assert ideation_context.metadata["chat_self_coding_code_task_id"] == (
            state.code_task_id
        )
        assert not transcript_store.path_for("code-task-old").exists()
        entries = [
            json.loads(line)
            for line in transcript_store.path_for(state.code_task_id)
            .read_text()
            .splitlines()
        ]
        assert [entry["kind"] for entry in entries] == [
            "user_message",
            "state_transition",
        ]
        assert entries[0]["text"] == "Никита: пересобери план"

    async def test_discussion_roundtrip_is_written_to_task_transcript(
        self, tmp_path: Path
    ) -> None:
        import json

        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState
        from src.skills.chat_self_coding.task_transcript import (
            FileTaskTranscriptStore,
        )

        transcript_store = FileTaskTranscriptStore(tmp_path)
        w = _build(
            classifier_intent="other",
            discussion_skill=_fake_skill_returning("да, обсудим аккуратно"),
            task_transcript_store=transcript_store,
        )
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.IDLE,
                code_task_id="code-task-fixed",
            )
        )

        await w.skill.execute("давай обсудим runtime", _ctx())

        raw_lines = (
            transcript_store.path_for("code-task-fixed").read_text().splitlines()
        )
        entries = [json.loads(line) for line in raw_lines]
        assert [entry["kind"] for entry in entries] == [
            "user_message",
            "assistant_message",
        ]
        assert entries[0]["text"] == "Никита: давай обсудим runtime"
        assert entries[1]["text"] == "Жвуша: да, обсудим аккуратно"

    async def test_merge_intent_runs_done_merge_handler(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="merge")
        await w.state_store.save(
            ChatSelfCodingState(
                user_id=1,
                stage=Stage.DONE,
                active_spec_slug="my-spec",
            )
        )

        result = await w.skill.execute("слей", _ctx())

        assert result.success
        w.merge_handler.assert_awaited_once()
        state = await w.state_store.load(1)
        assert state is not None
        assert state.stage == Stage.IDLE
        assert state.active_spec_slug is None


# ---------------------------------------------------------------------------
# Progress edit edge cases
# ---------------------------------------------------------------------------


class TestArchitectProgressEdits:
    async def test_telegram_unchanged_edit_is_ignored(self) -> None:
        from src.skills.chat_self_coding.skill import _edit_architect_progress

        bot = AsyncMock()
        bot.edit_message_text = AsyncMock(
            side_effect=TelegramBadRequest(
                method=EditMessageText(chat_id=1, message_id=42, text="x"),
                message=(
                    "message is not modified: specified new message content "
                    "and reply markup are exactly the same"
                ),
            )
        )

        await _edit_architect_progress(
            bot=bot,
            chat_id=1,
            message_id=42,
            percent=94,
            detail="План всё ещё проверяется. Я здесь, не зависла.",
        )

        bot.edit_message_text.assert_awaited_once()

    def test_waiting_progress_does_not_claim_near_done(self) -> None:
        from src.skills.chat_self_coding.skill import _ARCHITECT_PROGRESS_WAIT_PERCENT

        assert _ARCHITECT_PROGRESS_WAIT_PERCENT < 50


# ---------------------------------------------------------------------------
# Recent message tail
# ---------------------------------------------------------------------------


class TestRecentMessages:
    async def test_user_message_appended_to_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        w = _build(classifier_intent="other")
        await w.state_store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await w.skill.execute("привет", _ctx())
        state = await w.state_store.load(1)
        assert state is not None
        assert any("привет" in m for m in state.recent_messages)
