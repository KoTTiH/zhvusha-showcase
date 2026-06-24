"""Context pack builder tests for Agent Runtime jobs."""

from __future__ import annotations


def test_context_pack_builder_normalizes_inputs_and_keeps_recent_chat() -> None:
    from src.agent_runtime.context import ContextPackBuilder

    builder = ContextPackBuilder(max_chat_messages=2)

    pack = builder.build(
        user_request="  Сравни пост с проектом.  ",
        chat_context=("старое", "Никита: ща скину", "Никита: вот пост"),
        active_code_state=" discussion ",
        attachments=("attachments/post.png", ""),
        relevant_files=("src/bot/main.py",),
        constraints=("read-only", "read-only"),
    )

    assert pack.user_request == "Сравни пост с проектом."
    assert pack.chat_context == ("Никита: ща скину", "Никита: вот пост")
    assert pack.active_code_state == "discussion"
    assert pack.attachments == ("attachments/post.png",)
    assert pack.constraints == ("read-only",)


def test_context_pack_builder_fingerprint_is_stable_for_same_source_message() -> None:
    from src.agent_runtime.context import ContextPackBuilder

    builder = ContextPackBuilder()
    pack = builder.build(user_request="изучи")

    first = builder.fingerprint(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:42",
        kind="source_compare",
        context_pack=pack,
    )
    second = builder.fingerprint(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:42",
        kind="source_compare",
        context_pack=pack,
    )
    third = builder.fingerprint(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:43",
        kind="source_compare",
        context_pack=pack,
    )

    assert first == second
    assert first != third


def test_context_pack_builder_discovers_relevant_files_from_request(tmp_path) -> None:
    from src.agent_runtime.context import ContextPackBuilder
    from src.agent_runtime.retrieval import RelevantFileFinder

    (tmp_path / "src" / "agent_runtime").mkdir(parents=True)
    (tmp_path / "src" / "agent_runtime" / "rendering.py").write_text(
        "# renderer",
        encoding="utf-8",
    )
    (tmp_path / "src" / "skills" / "chat_self_coding").mkdir(parents=True)
    (tmp_path / "src" / "skills" / "chat_self_coding" / "attachments.py").write_text(
        "# attachments",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "other.md").write_text("other", encoding="utf-8")

    builder = ContextPackBuilder(
        relevant_file_finder=RelevantFileFinder(project_root=tmp_path),
        max_relevant_files=2,
    )

    pack = builder.build(
        user_request="проверь agent runtime renderer и самокодинг вложения",
    )

    assert set(pack.relevant_files) == {
        "src/agent_runtime/rendering.py",
        "src/skills/chat_self_coding/attachments.py",
    }


def test_context_pack_builder_preserves_explicit_relevant_files_first(
    tmp_path,
) -> None:
    from src.agent_runtime.context import ContextPackBuilder
    from src.agent_runtime.retrieval import RelevantFileFinder

    (tmp_path / "src" / "bot").mkdir(parents=True)
    (tmp_path / "src" / "bot" / "main.py").write_text("# bot", encoding="utf-8")
    (tmp_path / "src" / "agent_runtime").mkdir(parents=True)
    (tmp_path / "src" / "agent_runtime" / "runtime.py").write_text(
        "# runtime",
        encoding="utf-8",
    )

    builder = ContextPackBuilder(
        relevant_file_finder=RelevantFileFinder(project_root=tmp_path),
        max_relevant_files=3,
    )

    pack = builder.build(
        user_request="проверь src/bot/main.py и agent runtime",
        relevant_files=("src/bot/main.py",),
    )

    assert pack.relevant_files[0] == "src/bot/main.py"
    assert "src/agent_runtime/runtime.py" in pack.relevant_files
