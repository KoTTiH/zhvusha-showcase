from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Literal
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage
from src.bot import main as bot_main
from src.dialogue.state import DialogueState, FileDialogueStateStore
from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult


class _RequiredSkill(InlineSkill):
    name: ClassVar[str] = "required_gate_test"
    description: ClassVar[str] = "Required gate test"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.POSTS_TO_CHANNEL]

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/gate") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        self.executed.append(message)
        return SkillResult(success=True, response=f"executed: {message}")


class _NaturalRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "natural_required_gate_test"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 0.93 if message.startswith("опубликуй пост") else 0.0


class _MemoryStateStore:
    def __init__(self) -> None:
        self._kv: dict[int, object] = {}

    async def load(self, user_id: int) -> object | None:
        return self._kv.get(user_id)

    async def save(self, state: Any) -> None:
        self._kv[state.user_id] = state

    async def clear(self, user_id: int) -> None:
        self._kv.pop(user_id, None)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    bot_main._reset_chat_busy_state_for_tests()
    monkeypatch.setattr(bot_main, "_agent_runtime", None)
    monkeypatch.setattr(bot_main, "_source_compare_background_runner", None)


def _ctx(bot: object) -> AgentContext:
    return AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=1,
        bot=bot,
    )


async def test_dispatcher_does_not_execute_required_skill_without_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _RequiredSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    await bot_main._process_text_message("/gate publish", _ctx(bot))

    assert skill.executed == []
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert "Нужно решение" in sent_text
    assert "required_gate_test" in sent_text


async def test_dispatcher_does_not_execute_natural_required_skill_without_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _NaturalRequiredSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())

    await bot_main._process_text_message("опубликуй пост hello", _ctx(bot))

    assert skill.executed == []
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert "Нужно решение" in sent_text
    assert "natural_required_gate_test" in sent_text


async def test_dispatcher_executes_required_skill_after_text_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _RequiredSkill()
    monkeypatch.setattr(bot_main, "_skills", [skill])
    bot = SimpleNamespace(send_message=AsyncMock())
    ctx = _ctx(bot)

    await bot_main._process_text_message("/gate publish", ctx)
    await bot_main._process_text_message("да", ctx)

    assert skill.executed == ["/gate publish"]
    sent_texts = [call.kwargs["text"] for call in bot.send_message.await_args_list]
    assert any("executed: /gate publish" in text for text in sent_texts)


async def test_dispatcher_routes_confirmed_engineering_followup_to_code_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.skills.chat_self_coding.intent_classifier import (
        Intent,
        IntentClassification,
    )
    from src.skills.chat_self_coding.skill import ChatSelfCodingSkill

    FileDialogueStateStore(tmp_path).save(
        DialogueState(
            chat_id="12345",
            selected_skill="chat_response",
            last_user_message="сделай live browser link в телегу",
            last_assistant_response=(
                "делаю. беру за основу live browser link в телегу + "
                "persistent session, дальше надо врезать это в runtime и репу."
            ),
        )
    )
    monkeypatch.setattr(bot_main, "_workspace_root", lambda: tmp_path)

    classifier = AsyncMock(
        return_value=IntentClassification(intent=Intent.OTHER, confidence=0.9)
    )
    ideation = AsyncMock()
    ideation.execute = AsyncMock(
        return_value=SkillResult(
            success=True,
            response="",
            metadata={"slug": "live-browser-human-verification", "tier": 3},
        )
    )
    skill = ChatSelfCodingSkill(
        admin_user_id=12345,
        state_store=_MemoryStateStore(),  # type: ignore[arg-type]
        intent_classifier=classifier,
        ideation_skill=ideation,
        implement_skill=AsyncMock(),
        spec_skill=AsyncMock(),
    )
    monkeypatch.setattr(bot_main, "_skills", [skill])

    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        edit_message_text=AsyncMock(),
        send_photo=AsyncMock(),
    )

    await bot_main._process_text_message("Делай", _ctx(bot))

    ideation.execute.assert_awaited_once()
    sent_message = ideation.execute.await_args.args[0]
    assert sent_message.startswith("/spec_create ")
    assert "Диалог до входа в /код" in sent_message
    assert "live browser link" in sent_message


async def test_emit_skill_response_persists_telegram_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bot = SimpleNamespace(
        send_message=AsyncMock(
            side_effect=TelegramNetworkError(
                method=SendMessage(chat_id=12345, text="hello"),
                message="ConnectError after 5 attempts",
            )
        ),
        send_photo=AsyncMock(),
    )
    monkeypatch.setattr(bot_main, "_workspace_root", lambda: tmp_path)

    result = SkillResult(
        success=True,
        response="hello",
        metadata={"skill_name": "fake"},
    )

    response = await bot_main._emit_skill_response(result, _ctx(bot))

    assert response is None
    failure_dir = tmp_path / "runtime" / "telegram_delivery_failures"
    failures = list(failure_dir.glob("*.json"))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text(encoding="utf-8"))
    assert payload["chat_id"] == 12345
    assert payload["text"] == "hello"
    assert payload["parse_mode"] == "HTML"
    assert payload["error_type"] == "TelegramNetworkError"
    bot.send_photo.assert_not_awaited()


async def test_emit_skill_response_sends_image_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "agent_runtime" / "browser_artifacts" / "page.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        send_photo=AsyncMock(return_value=SimpleNamespace(message_id=2)),
    )
    monkeypatch.setattr(bot_main, "_workspace_root", lambda: tmp_path)

    result = SkillResult(
        success=True,
        response="готово",
        metadata={
            "skill_name": "web_research",
            "artifacts": ("agent_runtime/browser_artifacts/page.png",),
            "deliver_artifacts_to_chat": True,
        },
    )

    await bot_main._emit_skill_response(result, _ctx(bot))

    bot.send_message.assert_awaited_once()
    bot.send_photo.assert_awaited_once()
    sent_photo = bot.send_photo.await_args.kwargs["photo"]
    assert str(sent_photo.path) == str(artifact)


async def test_emit_skill_response_sends_image_artifact_named_in_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = (
        tmp_path
        / "agent_runtime"
        / "computer_use"
        / "screenshots"
        / "browser-screenshot-result.png"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n")
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        send_photo=AsyncMock(return_value=SimpleNamespace(message_id=2)),
    )
    monkeypatch.setattr(bot_main, "_workspace_root", lambda: tmp_path)

    result = SkillResult(
        success=True,
        response=(
            "новый скрин: "
            "agent_runtime/computer_use/screenshots/browser-screenshot-result.png"
        ),
        metadata={"skill_name": "chat_response"},
    )

    await bot_main._emit_skill_response(result, _ctx(bot))

    bot.send_message.assert_awaited_once()
    bot.send_photo.assert_awaited_once()
    sent_photo = bot.send_photo.await_args.kwargs["photo"]
    assert str(sent_photo.path) == str(artifact)
