from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.dialogue.state import (
    DialogueState,
    DialogueStatePatch,
    FileDialogueStateStore,
    render_dialogue_context,
    render_dialogue_status,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_state_patch_keeps_recipient_hint_without_executable_chat_id(
    tmp_path: Path,
) -> None:
    store = FileDialogueStateStore(tmp_path)

    state = store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            recipient_hint="Тоше",
            draft_message="Тош, не расстраивайся. Дота жива, пока ты живой.",
            missing_fields=("chat_id",),
            confidence=0.88,
            source="telegram_mcp_personal.prepare",
        ),
    )

    assert state.recipient_hint == "Тоше"
    assert state.executable_chat_id == ""
    assert state.draft_message == "Тош, не расстраивайся. Дота жива, пока ты живой."
    assert state.missing_fields == ("chat_id",)
    assert state.pending_action == "telegram_send"

    rendered = render_dialogue_context(
        state,
        decision_recent_messages=(
            "Собеседник: Не мне, а Тоше\nЖвуша: Не хватает @username/id для Тоше."
        ),
    )

    assert "pending_action: telegram_send" in rendered
    assert "recipient_hint: Тоше" in rendered
    assert "executable_chat_id: missing" in rendered
    assert "missing_fields: chat_id" in rendered
    assert "<RECENT_DECISION_CONTEXT>" in rendered
    assert '"recipient_hint"' not in rendered


def test_state_patch_can_clear_stale_executable_target(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)
    store.save(
        DialogueState(
            chat_id="12345",
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            recipient_hint="@KoTTiH",
            executable_chat_id="@KoTTiH",
            draft_message="старый текст",
        )
    )

    state = store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            recipient_hint="Тоше",
            draft_message="новый черновик",
            missing_fields=("chat_id",),
            clear_executable_chat_id=True,
            append_correction="Последнее 'пиши ему' относится к Тоше, не к @KoTTiH.",
        ),
    )

    assert state.recipient_hint == "Тоше"
    assert state.executable_chat_id == ""
    assert state.draft_message == "новый черновик"
    assert state.corrections == (
        "Последнее 'пиши ему' относится к Тоше, не к @KoTTiH.",
    )


def test_state_patch_missing_chat_id_clears_stale_executable_target(
    tmp_path: Path,
) -> None:
    store = FileDialogueStateStore(tmp_path)
    store.save(
        DialogueState(
            chat_id="12345",
            pending_action="telegram_send",
            recipient_hint="@KoTTiH",
            executable_chat_id="@KoTTiH",
            draft_message="старый текст",
        )
    )

    state = store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            recipient_hint="Тоше",
            missing_fields=("chat_id",),
            draft_message="новый черновик",
        ),
    )

    assert state.recipient_hint == "Тоше"
    assert state.missing_fields == ("chat_id",)
    assert state.executable_chat_id == ""
    assert "executable_chat_id: missing" in render_dialogue_context(state)
    assert "@KoTTiH" not in render_dialogue_context(state)


def test_loaded_state_conflict_prefers_missing_chat_id_over_executable_target(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "logs" / "777" / "dialogue_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "chat_id": "777",
                "pending_action": "telegram_send",
                "recipient_hint": "Тоше",
                "executable_chat_id": "@KoTTiH",
                "missing_fields": ["chat_id"],
                "draft_message": "новый черновик",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = FileDialogueStateStore(tmp_path).load("777")

    assert state.executable_chat_id == ""
    rendered = render_dialogue_context(state)
    assert "executable_chat_id: missing" in rendered
    assert "@KoTTiH" not in rendered


def test_state_patch_drops_human_hint_as_executable_chat_id(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)

    state = store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            recipient_hint="Тоше",
            executable_chat_id="Тоше",
            draft_message="Тош, проверь личку.",
            missing_fields=("chat_id",),
        ),
    )

    assert state.recipient_hint == "Тоше"
    assert state.executable_chat_id == ""
    assert state.missing_fields == ("chat_id",)

    username_state = store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(executable_chat_id="@Anroxa2748"),
    )

    assert username_state.executable_chat_id == "@Anroxa2748"


def test_file_store_quarantines_corrupt_state_and_recovers_empty(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "logs" / "777" / "dialogue_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{not json", encoding="utf-8")
    store = FileDialogueStateStore(tmp_path)

    state = store.load("777")

    assert state.chat_id == "777"
    assert state.has_signal() is False
    assert not state_path.exists()
    quarantined = list(state_path.parent.glob("dialogue_state.corrupt-*.json"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"


def test_file_store_persists_state_under_chat_log_dir(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)
    state = store.apply_patch(
        chat_id="777",
        patch=DialogueStatePatch(
            active_topic="личный Telegram",
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            recipient_hint="Тоше",
            missing_fields=("chat_id",),
        ),
    )

    path = tmp_path / "logs" / "777" / "dialogue_state.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["recipient_hint"] == "Тоше"

    loaded = store.load("777")
    assert loaded == state


def test_file_store_expires_stale_state_when_age_policy_enabled(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
    store = FileDialogueStateStore(
        tmp_path,
        max_age_seconds=3600,
        clock=lambda: now,
    )
    state_path = tmp_path / "logs" / "777" / "dialogue_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        DialogueState(
            chat_id="777",
            pending_action="telegram_send",
            recipient_hint="Тоше",
            updated_at=(now - timedelta(hours=2)).isoformat(),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    state = store.load("777")

    assert state.chat_id == "777"
    assert state.has_signal() is False
    assert not state_path.exists()
    expired = list(state_path.parent.glob("dialogue_state.expired-*.json"))
    assert len(expired) == 1
    assert "telegram_send" in expired[0].read_text(encoding="utf-8")


def test_file_store_keeps_recent_state_under_age_policy(tmp_path: Path) -> None:
    now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
    store = FileDialogueStateStore(
        tmp_path,
        max_age_seconds=3600,
        clock=lambda: now,
    )
    recent = DialogueState(
        chat_id="777",
        pending_action="telegram_send",
        recipient_hint="Тоше",
        updated_at=(now - timedelta(minutes=30)).isoformat(),
    )
    store.save(recent)

    loaded = store.load("777")

    assert loaded.pending_action == "telegram_send"
    assert loaded.recipient_hint == "Тоше"


def test_file_store_preserves_multi_turn_context_across_restarts(
    tmp_path: Path,
) -> None:
    first_process = FileDialogueStateStore(tmp_path)
    first_process.apply_patch(
        chat_id=777,
        patch=DialogueStatePatch(
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            recipient_hint="@KoTTiH",
            executable_chat_id="@KoTTiH",
            draft_message="старый черновик",
            increment_turn=True,
            source="turn-1",
        ),
    )

    second_process = FileDialogueStateStore(tmp_path)
    corrected = second_process.apply_patch(
        chat_id=777,
        patch=DialogueStatePatch(
            recipient_hint="Тоше",
            draft_message="Тош, не расстраивайся.",
            missing_fields=("chat_id",),
            append_correction="Не @KoTTiH, а Тоше.",
            increment_turn=True,
            source="turn-2",
        ),
    )

    assert corrected.turn_count == 2
    assert corrected.recipient_hint == "Тоше"
    assert corrected.executable_chat_id == ""
    assert corrected.missing_fields == ("chat_id",)

    third_process = FileDialogueStateStore(tmp_path)
    resolved = third_process.apply_patch(
        chat_id=777,
        patch=DialogueStatePatch(
            executable_chat_id="@Anroxa2748",
            clear_missing_fields=True,
            increment_turn=True,
            source="turn-3",
        ),
    )

    assert resolved.turn_count == 3
    assert resolved.recipient_hint == "Тоше"
    assert resolved.executable_chat_id == "@Anroxa2748"
    assert resolved.missing_fields == ()
    assert resolved.draft_message == "Тош, не расстраивайся."
    assert resolved.corrections == ("Не @KoTTiH, а Тоше.",)

    rendered = render_dialogue_context(resolved)
    assert "recipient_hint: Тоше" in rendered
    assert "executable_chat_id: @Anroxa2748" in rendered
    assert "missing_fields: chat_id" not in rendered
    assert "Не @KoTTiH, а Тоше." in rendered


def test_render_dialogue_status_summarizes_without_raw_text() -> None:
    state = DialogueState(
        chat_id="777",
        mode="personal",
        pending_action="telegram_send",
        selected_skill="telegram_mcp_personal",
        last_user_message="сырой пользовательский текст",
        last_assistant_response="сырой ответ Жвуши",
        recipient_hint="Тоше",
        draft_message="секретный черновик сообщения",
        missing_fields=("chat_id",),
        last_tool="telegram_mcp_send",
        last_result="failure",
        turn_count=4,
    )

    status = render_dialogue_status(state)

    assert "Диалоговая память" in status
    assert "mode: personal" in status
    assert "turn_count: 4" in status
    assert "pending_action: telegram_send" in status
    assert "recipient_hint: Тоше" in status
    assert "executable_chat_id: missing" in status
    assert "draft_message: present" in status
    assert "missing_fields: chat_id" in status
    assert "last_tool: telegram_mcp_send" in status
    assert "last_result: failure" in status
    assert "сырой пользовательский текст" not in status
    assert "сырой ответ Жвуши" not in status
    assert "секретный черновик сообщения" not in status


def test_render_dialogue_status_reports_empty_state() -> None:
    assert (
        render_dialogue_status(DialogueState(chat_id="777"))
        == "Диалоговая память пуста."
    )
