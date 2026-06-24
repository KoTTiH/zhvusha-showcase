from __future__ import annotations

from typing import TYPE_CHECKING

from src.dialogue.state import FileDialogueStateStore
from src.dialogue.updater import DialogueStateUpdater
from src.skills.base import SkillResult

if TYPE_CHECKING:
    from pathlib import Path


def test_updater_records_user_message_and_skill_result_patch(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)
    updater = DialogueStateUpdater(store)

    updater.record_user_message(chat_id=12345, text="Пиши ему", mode="personal")
    result = SkillResult(
        success=True,
        response="Не хватает @username/id для Тоше.",
        metadata={
            "skill_name": "telegram_mcp_personal",
            "dialogue_state_patch": {
                "pending_action": "telegram_send",
                "selected_skill": "telegram_mcp_personal",
                "recipient_hint": "Тоше",
                "draft_message": "Тош, не расстраивайся.",
                "missing_fields": ["chat_id"],
                "clear_executable_chat_id": True,
            },
        },
    )

    updater.record_skill_result(chat_id=12345, result=result)

    state = store.load(12345)
    assert state.last_user_message == "Пиши ему"
    assert state.last_assistant_response == "Не хватает @username/id для Тоше."
    assert state.turn_count == 1
    assert state.selected_skill == "telegram_mcp_personal"
    assert state.recipient_hint == "Тоше"
    assert state.executable_chat_id == ""
    assert state.missing_fields == ("chat_id",)


def test_updater_can_skip_body_layer_response_text(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)
    updater = DialogueStateUpdater(store)

    updater.record_skill_result(
        chat_id=12345,
        result=SkillResult(
            success=True,
            response="Не хватает @username/id для Тоше.",
            metadata={
                "skill_name": "telegram_mcp_personal",
                "skip_dialogue_assistant_response": True,
                "dialogue_state_patch": {
                    "pending_action": "telegram_send",
                    "recipient_hint": "Тоше",
                    "missing_fields": ["chat_id"],
                },
            },
        ),
    )

    state = store.load(12345)
    assert state.last_assistant_response == ""
    assert state.recipient_hint == "Тоше"
    assert state.missing_fields == ("chat_id",)


def test_updater_records_non_text_observation_as_structured_state(
    tmp_path: Path,
) -> None:
    store = FileDialogueStateStore(tmp_path)
    updater = DialogueStateUpdater(store)

    updater.record_observation(
        chat_id=12345,
        mode="personal",
        kind="photo_observation",
        summary="Пользователь прислал 2 фото с подписью.",
        source="photo_vision",
    )

    state = store.load(12345)
    assert state.turn_count == 1
    assert state.last_intent == "photo_observation"
    assert state.last_user_message == "Пользователь прислал 2 фото с подписью."
    assert state.last_tool == "photo_vision"
    assert state.last_result == "success"
    assert state.last_update_source == "bot.photo_observation"


def test_updater_extracts_personal_alias_candidates(tmp_path: Path) -> None:
    from src.dialogue.people import FilePeopleAliasStore

    store = FileDialogueStateStore(tmp_path)
    people_alias_store = FilePeopleAliasStore(tmp_path)
    updater = DialogueStateUpdater(store, people_alias_store=people_alias_store)

    updater.record_user_message(
        chat_id=12345,
        text="@Anroxa2748 это Тоша",
        mode="personal",
        source_message_id="tg:55",
    )

    result = people_alias_store.lookup(chat_id=12345, alias="Тоше")
    assert result.status == "needs_confirmation"
    assert result.suggested_recipient == "@Anroxa2748"
    assert result.can_execute is False
    assert result.candidates[0].source_message_id == "tg:55"
    assert result.candidates[0].scope == "personal"


def test_updater_does_not_extract_aliases_outside_personal_mode(
    tmp_path: Path,
) -> None:
    from src.dialogue.people import FilePeopleAliasStore

    store = FileDialogueStateStore(tmp_path)
    people_alias_store = FilePeopleAliasStore(tmp_path)
    updater = DialogueStateUpdater(store, people_alias_store=people_alias_store)

    updater.record_user_message(
        chat_id=12345,
        text="@Anroxa2748 это Тоша",
        mode="social",
        source_message_id="tg:55",
    )

    result = people_alias_store.lookup(chat_id=12345, alias="Тоше")
    assert result.status == "not_found"
