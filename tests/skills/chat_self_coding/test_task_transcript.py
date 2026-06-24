"""Durable per-task transcript for /код sessions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock


async def test_file_task_transcript_store_appends_jsonl_entries(
    tmp_path: Path,
) -> None:
    from src.skills.chat_self_coding.task_transcript import FileTaskTranscriptStore

    store = FileTaskTranscriptStore(tmp_path)

    await store.append(
        task_id="code-task-fixed",
        user_id=1,
        kind="user_message",
        text="Никита: делай",
        slug="my-spec",
        payload={"stage": "discussion"},
    )

    path = store.path_for("code-task-fixed")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    entry = json.loads(raw_lines[0])
    assert entry["task_id"] == "code-task-fixed"
    assert entry["user_id"] == 1
    assert entry["kind"] == "user_message"
    assert entry["text"] == "Никита: делай"
    assert entry["slug"] == "my-spec"
    assert entry["payload"] == {"stage": "discussion"}

    report = store.render_markdown("code-task-fixed")
    assert "# /код task transcript" in report
    assert "code-task-fixed" in report
    assert "Никита: делай" in report


async def test_transcript_block_publisher_records_task_events(
    tmp_path: Path,
) -> None:
    from src.skills.chat_self_coding.events import BlockEvent, BlockEventType
    from src.skills.chat_self_coding.task_transcript import (
        FileTaskTranscriptStore,
        TranscriptBlockPublisher,
    )

    delegate = AsyncMock()
    store = FileTaskTranscriptStore(tmp_path)
    publisher = TranscriptBlockPublisher(delegate=delegate, transcript_store=store)
    event = BlockEvent(
        user_id=1,
        event_type=BlockEventType.IMPLEMENTATION,
        slug="my-spec",
        task_id="code-task-fixed",
        payload={"detail": "Патчу bridge.", "stage": "реализация"},
    )

    await publisher.publish(event)

    delegate.publish.assert_awaited_once_with(event)
    entry = json.loads(store.path_for("code-task-fixed").read_text().splitlines()[0])
    assert entry["kind"] == "block_event"
    assert entry["text"] == "implementation: Патчу bridge."
    assert entry["slug"] == "my-spec"
    assert entry["payload"]["event_type"] == "implementation"
    assert entry["payload"]["stage"] == "реализация"


async def test_transcript_block_publisher_skips_events_without_task_id(
    tmp_path: Path,
) -> None:
    from src.skills.chat_self_coding.events import BlockEvent, BlockEventType
    from src.skills.chat_self_coding.task_transcript import (
        FileTaskTranscriptStore,
        TranscriptBlockPublisher,
    )

    delegate = AsyncMock()
    store = FileTaskTranscriptStore(tmp_path)
    publisher = TranscriptBlockPublisher(delegate=delegate, transcript_store=store)
    event = BlockEvent(
        user_id=1,
        event_type=BlockEventType.PLAN,
        slug="legacy-spec",
        payload={"summary": "План без chat task."},
    )

    await publisher.publish(event)

    delegate.publish.assert_awaited_once_with(event)
    assert list(tmp_path.glob("*.jsonl")) == []
