from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.skills.chat_response.context_loader import ContextLoader


def _setup_workspace(root: Path) -> None:
    """Create a minimal workspace for testing."""
    (root / "personality").mkdir(parents=True)
    (root / "personality" / "core.md").write_text("I am Zhvusha.")
    (root / "personality" / "genes.md").write_text("Curiosity: HIGH")
    (root / "diary").mkdir(parents=True)
    (root / "diary" / "2026-03-31.md").write_text("Today was interesting.")
    (root / "memory" / "people").mkdir(parents=True)


def test_load_personality_only_personality_dir(tmp_path: Path):
    """load_personality returns only personality/* files, not diary."""
    _setup_workspace(tmp_path)
    loader = ContextLoader(tmp_path)
    result = loader.load_personality()

    assert "I am Zhvusha." in result
    assert "Curiosity: HIGH" in result
    assert "Today was interesting." not in result


# --- Staging directory (.staging/learnings_immediate.md) support ---


def _setup_staging(root: Path, content: str) -> Path:
    staging_dir = root / "personality" / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_file = staging_dir / "learnings_immediate.md"
    staging_file.write_text(content, encoding="utf-8")
    return staging_file


def test_load_personality_includes_staging_immediate_when_present(
    tmp_path: Path,
) -> None:
    _setup_workspace(tmp_path)
    _setup_staging(tmp_path, "## [rule] tone\n**Statement:** не писать формально\n")
    loader = ContextLoader(tmp_path)

    result = loader.load_personality()

    assert "не писать формально" in result
    assert "[rule] tone" in result


def test_load_personality_places_staging_after_core(tmp_path: Path) -> None:
    """Recency bias: staging must appear AFTER core.md so last-in-prompt
    weight favors recent learnings over base personality."""
    _setup_workspace(tmp_path)
    _setup_staging(tmp_path, "STAGING_MARKER staging content")
    loader = ContextLoader(tmp_path)

    result = loader.load_personality()

    core_idx = result.index("I am Zhvusha.")
    staging_idx = result.index("STAGING_MARKER")
    assert staging_idx > core_idx, (
        "staging learnings must land after core.md in system prompt "
        "(recency bias — last content wins)"
    )


def test_load_personality_no_crash_when_staging_missing(tmp_path: Path) -> None:
    _setup_workspace(tmp_path)  # no .staging/ directory
    loader = ContextLoader(tmp_path)

    result = loader.load_personality()

    assert "I am Zhvusha." in result
    assert ".staging" not in result


def test_load_personality_skips_empty_staging_file(tmp_path: Path) -> None:
    """Empty staging file must not produce an orphan '### path' header."""
    _setup_workspace(tmp_path)
    _setup_staging(tmp_path, "   \n\n  ")  # whitespace only
    loader = ContextLoader(tmp_path)

    result = loader.load_personality()

    assert "learnings_immediate.md" not in result
    assert "I am Zhvusha." in result


def test_load_personality_does_not_leak_staging_pending(tmp_path: Path) -> None:
    """Only learnings_immediate.md is loaded. learnings_pending.md is reserved
    for /morning review and must not reach the system prompt."""
    _setup_workspace(tmp_path)
    staging_dir = tmp_path / "personality" / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "learnings_immediate.md").write_text(
        "IMMEDIATE_MARKER", encoding="utf-8"
    )
    (staging_dir / "learnings_pending.md").write_text(
        "PENDING_MARKER", encoding="utf-8"
    )
    loader = ContextLoader(tmp_path)

    result = loader.load_personality()

    assert "IMMEDIATE_MARKER" in result
    assert "PENDING_MARKER" not in result


def test_load_personality_personal_includes_nested_voice_samples(
    tmp_path: Path,
) -> None:
    _setup_workspace(tmp_path)
    sample_dir = tmp_path / "personality" / "voice_samples" / "diary"
    sample_dir.mkdir(parents=True)
    (sample_dir / "example.md").write_text(
        "VOICE_SAMPLE_MARKER\nживой старый голос",
        encoding="utf-8",
    )
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="personal")

    assert "### personality/voice_samples/diary/example.md" in result
    assert "VOICE_SAMPLE_MARKER" in result


def test_load_personality_assistant_skips_private_voice_samples(
    tmp_path: Path,
) -> None:
    _setup_workspace(tmp_path)
    sample_dir = tmp_path / "personality" / "voice_samples" / "diary"
    sample_dir.mkdir(parents=True)
    (sample_dir / "example.md").write_text(
        "PRIVATE_VOICE_SAMPLE_MARKER",
        encoding="utf-8",
    )
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="assistant")

    assert "PRIVATE_VOICE_SAMPLE_MARKER" not in result


def test_personal_loads_all(tmp_path: Path):
    _setup_workspace(tmp_path)
    loader = ContextLoader(tmp_path)
    result = loader.load_for_mode("personal")

    assert "I am Zhvusha." in result
    assert "Curiosity: HIGH" in result
    assert "Today was interesting." in result


def test_social_loads_only_personality(tmp_path: Path):
    _setup_workspace(tmp_path)
    loader = ContextLoader(tmp_path)
    result = loader.load_for_mode("social")

    assert "I am Zhvusha." in result
    assert "Curiosity: HIGH" in result
    assert "Today was interesting." not in result


def test_assistant_loads_only_personality(tmp_path: Path):
    _setup_workspace(tmp_path)
    loader = ContextLoader(tmp_path)
    result = loader.load_for_mode("assistant")

    assert "I am Zhvusha." in result
    assert "Curiosity: HIGH" in result
    assert "Today was interesting." not in result


def test_missing_files_no_crash(tmp_path: Path):
    # Empty workspace — no files exist
    loader = ContextLoader(tmp_path)
    result = loader.load_for_mode("personal")
    assert result == ""


def test_load_recent_messages(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [
        {"role": "user", "text": "привет"},
        {"role": "assistant", "text": "Привет, Никита!"},
        {"role": "user", "text": "как дела?"},
    ]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345)

    assert "Собеседник: привет" in result
    assert "Жвуша: Привет, Никита!" in result
    assert "Собеседник: как дела?" in result


def test_load_recent_messages_personal_max_20(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [{"role": "user", "text": f"msg {i}"} for i in range(25)]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345, mode="personal")

    assert "msg 5" in result
    assert "msg 24" in result
    assert "msg 4" not in result


def test_load_recent_messages_accepts_decision_window_limit(tmp_path: Path) -> None:
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [{"role": "user", "text": f"msg {i}"} for i in range(10)]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345, mode="personal", limit=5)

    assert "msg 5" in result
    assert "msg 9" in result
    assert "msg 4" not in result


def test_load_recent_messages_skips_vscode_codex_transport_probe(
    tmp_path: Path,
) -> None:
    chat_dir = tmp_path / "logs" / "vscode"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [
        {
            "role": "user",
            "source": "vscode",
            "source_actor": "codex",
            "text": "Codex проверяет bridge",
        },
        {
            "role": "assistant",
            "source": "vscode",
            "source_actor": "zhvusha",
            "text": "Связь есть, я в VS Code-чате.",
        },
        {
            "role": "user",
            "source": "vscode",
            "source_actor": "user",
            "text": "как дела у тебя?",
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
        encoding="utf-8",
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id="vscode", mode="personal")

    assert "как дела у тебя?" in result
    assert "Codex проверяет bridge" not in result
    assert "Связь есть, я в VS Code-чате" not in result


def test_load_recent_messages_keeps_non_probe_codex_dialogue(
    tmp_path: Path,
) -> None:
    chat_dir = tmp_path / "logs" / "vscode"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [
        {
            "role": "user",
            "source": "vscode",
            "source_actor": "codex",
            "text": "Жвуша, оцени context routing как dev helper.",
        },
        {
            "role": "assistant",
            "source": "vscode",
            "source_actor": "zhvusha",
            "reply_to_source_actor": "codex",
            "text": "Я бы оставила identity kernel всегда.",
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
        encoding="utf-8",
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id="vscode", mode="personal")

    assert "Codex: Жвуша, оцени context routing" in result
    assert "Я бы оставила identity kernel всегда" in result


def test_load_recent_messages_spans_previous_days(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)

    old_entries = [
        {"role": "user", "text": "старый контекст"},
        {"role": "assistant", "text": "прошлый ответ"},
    ]
    new_entries = [{"role": "user", "text": "новый вопрос"}]
    (chat_dir / "chat_2026-05-05.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in old_entries),
    )
    (chat_dir / "chat_2026-05-06.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in new_entries),
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345, mode="personal")

    assert "Собеседник: старый контекст" in result
    assert "Жвуша: прошлый ответ" in result
    assert "Собеседник: новый вопрос" in result
    assert result.index("старый контекст") < result.index("новый вопрос")


def test_load_recent_messages_assistant_max_15(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [{"role": "user", "text": f"msg {i}"} for i in range(20)]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345, mode="assistant")

    assert "msg 5" in result
    assert "msg 19" in result
    assert "msg 0" not in result
    assert "msg 4" not in result


def test_load_recent_messages_no_log(tmp_path: Path):
    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345)
    assert result == ""


def test_load_recent_messages_no_chat_id(tmp_path: Path):
    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages()
    assert result == ""


# --- Photo entries in recent messages ---


def test_load_recent_messages_photo_entry(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entry = {
        "role": "user",
        "text": "какой тебе больше подходит?",
        "photo_paths": ["media/2026-04-01_456_0.jpg", "media/2026-04-01_456_1.jpg"],
        "photo_description": "стилизованный крокодил с живыми глазами и улыбкой",
    }
    log_file.write_text(json.dumps(entry, ensure_ascii=False) + "\n")

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345)

    assert "какой тебе больше подходит?" in result
    assert "[фото: media/2026-04-01_456_0.jpg, media/2026-04-01_456_1.jpg]" in result
    assert "[на фото: стилизованный крокодил" in result


def test_load_recent_messages_photo_no_caption(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entry = {
        "role": "user",
        "text": "",
        "photo_paths": ["media/2026-04-01_42_0.jpg"],
        "photo_description": "скриншот кода",
    }
    log_file.write_text(json.dumps(entry, ensure_ascii=False) + "\n")

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345)

    assert "Собеседник:" in result
    assert "[фото: media/2026-04-01_42_0.jpg]" in result
    assert "[на фото: скриншот кода]" in result
    # No caption — label alone without text
    assert "Собеседник: \n" not in result


def test_load_recent_messages_mixed(tmp_path: Path):
    chat_dir = tmp_path / "logs" / "12345"
    chat_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_file = chat_dir / f"chat_{today}.jsonl"

    entries = [
        {"role": "user", "text": "привет"},
        {
            "role": "user",
            "text": "вот фотка",
            "photo_paths": ["media/img.jpg"],
            "photo_description": "котик",
        },
        {"role": "assistant", "text": "Милый котик!"},
    ]
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
    )

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=12345)

    assert "Собеседник: привет" in result
    assert "[фото: media/img.jpg]" in result
    assert "[на фото: котик]" in result
    assert "Жвуша: Милый котик!" in result


# --- Channel posts ---


def test_load_channel_posts_all(tmp_path: Path):
    posts_dir = tmp_path / "channel" / "posts"
    posts_dir.mkdir(parents=True)
    for i in range(1, 8):
        (posts_dir / f"2026-04-{i:02d}_1.md").write_text(f"Post {i}")

    loader = ContextLoader(tmp_path)
    result = loader.load_channel_posts()

    # All 7 posts loaded
    for i in range(1, 8):
        assert f"Post {i}" in result


def test_load_channel_posts_with_limit(tmp_path: Path):
    posts_dir = tmp_path / "channel" / "posts"
    posts_dir.mkdir(parents=True)
    for i in range(1, 8):
        (posts_dir / f"2026-04-{i:02d}_1.md").write_text(f"Post {i}")

    loader = ContextLoader(tmp_path)
    result = loader.load_channel_posts(limit=3)

    assert "Post 5" in result
    assert "Post 7" in result
    assert "Post 4" not in result


def test_load_channel_posts_with_since(tmp_path: Path):
    posts_dir = tmp_path / "channel" / "posts"
    posts_dir.mkdir(parents=True)
    (posts_dir / "2026-03-20_1.md").write_text("Old post")
    (posts_dir / "2026-04-01_1.md").write_text("Recent post")
    (posts_dir / "2026-04-05_1.md").write_text("Latest post")

    loader = ContextLoader(tmp_path)
    result = loader.load_channel_posts(since=date(2026, 3, 25))

    assert "Old post" not in result
    assert "Recent post" in result
    assert "Latest post" in result


def test_load_channel_posts_empty(tmp_path: Path):
    loader = ContextLoader(tmp_path)
    result = loader.load_channel_posts()
    assert result == ""


def test_load_channel_posts_chronological_order(tmp_path: Path):
    posts_dir = tmp_path / "channel" / "posts"
    posts_dir.mkdir(parents=True)
    (posts_dir / "2026-04-01_1.md").write_text("First")
    (posts_dir / "2026-04-02_1.md").write_text("Second")

    loader = ContextLoader(tmp_path)
    result = loader.load_channel_posts()

    first_pos = result.index("First")
    second_pos = result.index("Second")
    assert first_pos < second_pos


# --- Deduplication tests ---


def test_load_recent_messages_excludes_current(tmp_path: Path):
    """Current user message is excluded from recent messages."""
    log_dir = tmp_path / "logs" / "123"
    log_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines = [
        json.dumps({"role": "user", "text": "привет"}),
        json.dumps({"role": "assistant", "text": "Привет, Никита!"}),
        json.dumps({"role": "user", "text": "как дела?"}),
    ]
    (log_dir / f"chat_{today}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=123, exclude_text="как дела?")

    # "как дела?" should be excluded (it's the current message)
    assert "как дела?" not in result
    # Other messages should remain
    assert "привет" in result
    assert "Привет, Никита!" in result


def test_load_recent_messages_keeps_assistant_same_text(tmp_path: Path):
    """Assistant message with same text is NOT excluded."""
    log_dir = tmp_path / "logs" / "123"
    log_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines = [
        json.dumps({"role": "user", "text": "привет"}),
        json.dumps({"role": "assistant", "text": "повтори за мной"}),
    ]
    (log_dir / f"chat_{today}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=123, exclude_text="повтори за мной")

    # Assistant message should still be there (dedup only affects user messages)
    assert "повтори за мной" in result


def test_load_recent_messages_no_exclude_keeps_all(tmp_path: Path):
    """Without exclude_text, all messages are returned."""
    log_dir = tmp_path / "logs" / "123"
    log_dir.mkdir(parents=True)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines = [
        json.dumps({"role": "user", "text": "hello"}),
    ]
    (log_dir / f"chat_{today}.jsonl").write_text("\n".join(lines), encoding="utf-8")

    loader = ContextLoader(tmp_path)
    result = loader.load_recent_messages(chat_id=123)

    assert "hello" in result
