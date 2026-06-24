from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.bot.main import _dialogue_status_reply, _with_dialogue_context_metadata
from src.dialogue.state import DialogueState, DialogueStatePatch, FileDialogueStateStore
from src.skills.base import AgentContext

if TYPE_CHECKING:
    from pathlib import Path


def _write_chat_log(root: Path, chat_id: int, entries: list[dict[str, object]]) -> None:
    chat_dir = root / "logs" / str(chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    (chat_dir / f"chat_{today}.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries),
        encoding="utf-8",
    )


def test_dialogue_context_metadata_contains_state_and_two_recent_windows(
    tmp_path: Path,
) -> None:
    store = FileDialogueStateStore(tmp_path)
    store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            recipient_hint="Тоше",
            draft_message="Тош, не расстраивайся.",
            missing_fields=("chat_id",),
        ),
    )
    _write_chat_log(
        tmp_path,
        12345,
        [{"role": "user", "text": f"msg {index}"} for index in range(10)],
    )
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")

    updated = _with_dialogue_context_metadata(
        "Пиши ему",
        context,
        workspace_root=tmp_path,
    )

    metadata = updated.metadata
    assert metadata["dialogue_state"]["recipient_hint"] == "Тоше"
    assert metadata["dialogue_state"]["draft_message"] == "Тош, не расстраивайся."
    assert "recipient_hint: Тоше" in metadata["dialogue_context"]
    assert "draft_message: Тош, не расстраивайся." in metadata["dialogue_context"]

    assert "msg 0" in metadata["recent_messages"]
    assert "msg 9" in metadata["recent_messages"]
    assert "msg 4" not in metadata["recent_decision_messages"]
    assert "msg 5" in metadata["recent_decision_messages"]
    assert "msg 9" in metadata["recent_decision_messages"]


def test_dialogue_context_metadata_expires_stale_action_state(
    tmp_path: Path,
) -> None:
    stale = DialogueState(
        chat_id="12345",
        pending_action="telegram_send",
        selected_skill="telegram_mcp_personal",
        recipient_hint="@Anroxa2748",
        executable_chat_id="@Anroxa2748",
        draft_message="расскумаримся в дотке сегодня",
        last_tool="telegram_mcp_send",
        last_result="success",
        updated_at=(datetime.now(tz=UTC) - timedelta(days=7)).isoformat(),
    )
    FileDialogueStateStore(tmp_path).save(stale)
    context = AgentContext(user_id=12345, chat_id=12345, mode="personal")

    updated = _with_dialogue_context_metadata(
        "Открой статью и пришли скриншот",
        context,
        workspace_root=tmp_path,
    )

    metadata = updated.metadata
    assert metadata["dialogue_state"]["recipient_hint"] == ""
    assert metadata["dialogue_state"]["draft_message"] == ""
    assert "dialogue_context" not in metadata
    expired = list((tmp_path / "logs" / "12345").glob("dialogue_state.expired-*.json"))
    assert len(expired) == 1
    assert "расскумаримся в дотке сегодня" in expired[0].read_text(encoding="utf-8")


def test_dialogue_status_command_renders_safe_state_for_admin(tmp_path: Path) -> None:
    store = FileDialogueStateStore(tmp_path)
    store.apply_patch(
        chat_id=12345,
        patch=DialogueStatePatch(
            pending_action="telegram_send",
            selected_skill="telegram_mcp_personal",
            last_user_message="сырой пользовательский текст",
            last_assistant_response="сырой ответ",
            recipient_hint="Тоше",
            draft_message="секретный черновик",
            missing_fields=("chat_id",),
            source="test",
        ),
    )

    status = _dialogue_status_reply(
        "/dialogue_status",
        AgentContext(user_id=12345, chat_id=12345, mode="personal"),
        admin_user_id=12345,
        workspace_root=tmp_path,
    )

    assert status is not None
    assert "Диалоговая память" in status
    assert "pending_action: telegram_send" in status
    assert "recipient_hint: Тоше" in status
    assert "draft_message: present" in status
    assert "сырой пользовательский текст" not in status
    assert "сырой ответ" not in status
    assert "секретный черновик" not in status


def test_dialogue_status_command_is_admin_only(tmp_path: Path) -> None:
    status = _dialogue_status_reply(
        "/dialogue_status",
        AgentContext(user_id=100, chat_id=12345, mode="personal"),
        admin_user_id=12345,
        workspace_root=tmp_path,
    )

    assert status == "Эта команда доступна только Никите."
