"""Tests for chat_response/tools.py — tool definitions and execute_tool.

Covers:
- Tool definitions schema validity
- execute_tool routing and error handling
- _read_workspace_file: path traversal, missing file, empty, truncation
- _search_knowledge: empty query, no store, no results
- post_to_channel: send message via bot, archive to workspace
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.skills.chat_response.tools import (
    MAX_TOOL_CALLS,
    execute_tool,
    get_chat_tools,
)

# --- get_chat_tools ---


def test_chat_tools_schema() -> None:
    """Tool definitions have required fields."""
    tools = get_chat_tools()
    assert len(tools) == 10
    names = {t.name for t in tools}
    assert names == {
        "search_knowledge",
        "get_entry_content",
        "browse_categories",
        "read_workspace_file",
        "list_workspace",
        "read_project_file",
        "list_project_files",
        "search_project",
        "add_knowledge",
        "post_to_channel",
    }
    for t in tools:
        assert t.description
        assert t.input_schema["type"] == "object"


def test_operator_recovery_tool_hidden_by_default() -> None:
    names = {tool.name for tool in get_chat_tools()}

    assert "rollback_morning_personality_tail" not in names


def test_operator_recovery_tool_visible_for_vscode_codex_context() -> None:
    names = {
        tool.name
        for tool in get_chat_tools(
            {"source": "vscode", "source_actor": "codex", "chat_log_id": "vscode"}
        )
    }

    assert "rollback_morning_personality_tail" in names


def test_computer_use_tool_visible_only_when_enabled() -> None:
    default_names = {tool.name for tool in get_chat_tools()}
    enabled_tools = get_chat_tools({"computer_use_tool_enabled": True})
    enabled_names = {tool.name for tool in enabled_tools}

    assert "computer_use" not in default_names
    assert "computer_use" in enabled_names
    computer_tool = next(tool for tool in enabled_tools if tool.name == "computer_use")
    assert computer_tool.input_schema["required"] == ["action"]
    assert "artifact_requirements" in computer_tool.input_schema["properties"]


def test_max_tool_calls_constant() -> None:
    assert MAX_TOOL_CALLS == 5


# --- read_workspace_file ---


@pytest.mark.asyncio
async def test_read_workspace_file_success(tmp_path: Path) -> None:
    """Reads a file within workspace."""
    (tmp_path / "diary").mkdir()
    target = tmp_path / "diary" / "today.md"
    target.write_text("hello world", encoding="utf-8")

    result = await execute_tool(
        "read_workspace_file",
        {"path": "diary/today.md"},
        workspace_root=tmp_path,
    )
    assert "hello world" in result


@pytest.mark.asyncio
async def test_read_workspace_file_path_traversal(tmp_path: Path) -> None:
    """Path traversal is blocked by safe_path."""
    # Create a file outside workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret data", encoding="utf-8")

    ws = tmp_path / "workspace"
    ws.mkdir()

    result = await execute_tool(
        "read_workspace_file",
        {"path": "../outside/secret.txt"},
        workspace_root=ws,
    )
    assert "запрещён" in result.lower() or "не найден" in result.lower()


@pytest.mark.asyncio
async def test_read_workspace_file_prefix_collision(tmp_path: Path) -> None:
    """Path with shared prefix but different dir is blocked.

    E.g., workspace=/home/ws, path=../ws-evil/file → must be rejected.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    evil = tmp_path / "ws-evil"
    evil.mkdir()
    (evil / "steal.txt").write_text("stolen", encoding="utf-8")

    result = await execute_tool(
        "read_workspace_file",
        {"path": "../ws-evil/steal.txt"},
        workspace_root=ws,
    )
    assert "запрещён" in result.lower() or "не найден" in result.lower()
    assert "stolen" not in result


@pytest.mark.asyncio
async def test_read_workspace_file_missing(tmp_path: Path) -> None:
    result = await execute_tool(
        "read_workspace_file",
        {"path": "nonexistent.md"},
        workspace_root=tmp_path,
    )
    assert "не найден" in result.lower()


@pytest.mark.asyncio
async def test_read_workspace_file_empty_path(tmp_path: Path) -> None:
    result = await execute_tool(
        "read_workspace_file",
        {"path": ""},
        workspace_root=tmp_path,
    )
    assert "пустой" in result.lower()


@pytest.mark.asyncio
async def test_read_workspace_file_no_workspace() -> None:
    result = await execute_tool(
        "read_workspace_file",
        {"path": "file.md"},
        workspace_root=None,
    )
    assert "не настроен" in result.lower()


@pytest.mark.asyncio
async def test_read_workspace_file_truncation(tmp_path: Path) -> None:
    """Large files are truncated to 4000 chars."""
    big = tmp_path / "big.md"
    big.write_text("x" * 5000, encoding="utf-8")

    result = await execute_tool(
        "read_workspace_file",
        {"path": "big.md"},
        workspace_root=tmp_path,
    )
    assert "обрезано" in result
    assert len(result) < 5000


@pytest.mark.asyncio
async def test_read_workspace_file_reports_binary_artifact_exists(
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
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n\xff")

    result = await execute_tool(
        "read_workspace_file",
        {
            "path": "agent_runtime/computer_use/screenshots/browser-screenshot-result.png"
        },
        workspace_root=tmp_path,
    )

    assert "Бинарный workspace-файл найден" in result
    assert "слой доставки прикрепит файл" in result


# --- search_knowledge ---


@pytest.mark.asyncio
async def test_search_knowledge_empty_query() -> None:
    result = await execute_tool(
        "search_knowledge",
        {"query": ""},
        knowledge_store=AsyncMock(),
    )
    assert "пустой" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_no_store() -> None:
    result = await execute_tool(
        "search_knowledge",
        {"query": "test"},
        knowledge_store=None,
    )
    assert "не подключена" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_no_results() -> None:
    store = AsyncMock()
    store.hybrid_search = AsyncMock(return_value=[])

    result = await execute_tool(
        "search_knowledge",
        {"query": "nonexistent topic"},
        knowledge_store=store,
    )
    assert "не найдено" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_with_results() -> None:
    """Results are formatted with title and summary."""
    mock_result = AsyncMock()
    mock_result.id = 1

    mock_summary = AsyncMock()
    mock_summary.title = "Test Article"
    mock_summary.summary = "A test summary"

    store = AsyncMock()
    store.hybrid_search = AsyncMock(return_value=[mock_result])
    store.get_summaries = AsyncMock(return_value=[mock_summary])

    result = await execute_tool(
        "search_knowledge",
        {"query": "test"},
        knowledge_store=store,
    )
    assert "Test Article" in result
    assert "A test summary" in result


# --- unknown tool ---


@pytest.mark.asyncio
async def test_execute_unknown_tool() -> None:
    result = await execute_tool("nonexistent_tool", {})
    assert "неизвестный" in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_exception_returns_error() -> None:
    """Exception inside tool handler returns error string, never raises."""
    store = AsyncMock()
    store.hybrid_search = AsyncMock(side_effect=RuntimeError("db crash"))

    result = await execute_tool(
        "search_knowledge",
        {"query": "test"},
        knowledge_store=store,
    )
    assert "ошибка" in result.lower()


# --- get_entry_content ---


@pytest.mark.asyncio
async def test_get_entry_content_no_store() -> None:
    result = await execute_tool(
        "get_entry_content",
        {"entry_id": 1},
        knowledge_store=None,
    )
    assert "не подключена" in result.lower()


@pytest.mark.asyncio
async def test_get_entry_content_no_id() -> None:
    result = await execute_tool(
        "get_entry_content",
        {"entry_id": 0},
        knowledge_store=AsyncMock(),
    )
    assert "не указан" in result.lower()


@pytest.mark.asyncio
async def test_get_entry_content_not_found() -> None:
    store = AsyncMock()
    store.get_full_content = AsyncMock(return_value=None)

    result = await execute_tool(
        "get_entry_content",
        {"entry_id": 999},
        knowledge_store=store,
    )
    assert "не найдена" in result.lower()


@pytest.mark.asyncio
async def test_get_entry_content_found() -> None:
    entry = AsyncMock()
    entry.title = "Test Title"
    entry.tags = ["python", "ai"]
    entry.category_name_ru = "Инструменты"
    entry.content = "Full content here."

    store = AsyncMock()
    store.get_full_content = AsyncMock(return_value=entry)

    result = await execute_tool(
        "get_entry_content",
        {"entry_id": 1},
        knowledge_store=store,
    )
    assert "Test Title" in result
    assert "python" in result
    assert "Инструменты" in result
    assert "Full content here." in result


@pytest.mark.asyncio
async def test_get_entry_content_truncation() -> None:
    entry = AsyncMock()
    entry.title = "Big"
    entry.tags = []
    entry.category_name_ru = ""
    entry.content = "x" * 7000

    store = AsyncMock()
    store.get_full_content = AsyncMock(return_value=entry)

    result = await execute_tool(
        "get_entry_content",
        {"entry_id": 1},
        knowledge_store=store,
    )
    assert "обрезано" in result
    assert len(result) < 7000


# --- browse_categories ---


@pytest.mark.asyncio
async def test_browse_categories_no_store() -> None:
    result = await execute_tool("browse_categories", {}, knowledge_store=None)
    assert "не подключена" in result.lower()


@pytest.mark.asyncio
async def test_browse_categories_empty() -> None:
    store = AsyncMock()
    store.browse_categories = AsyncMock(return_value=[])

    result = await execute_tool("browse_categories", {}, knowledge_store=store)
    assert "нет" in result.lower()


@pytest.mark.asyncio
async def test_browse_categories_with_results() -> None:
    cat = AsyncMock()
    cat.name_ru = "Инструменты"
    cat.path = "tools"
    cat.entry_count = 5

    store = AsyncMock()
    store.browse_categories = AsyncMock(return_value=[cat])

    result = await execute_tool("browse_categories", {}, knowledge_store=store)
    assert "Инструменты" in result
    assert "tools" in result
    assert "5" in result


# --- list_workspace ---


@pytest.mark.asyncio
async def test_list_workspace_no_workspace() -> None:
    result = await execute_tool("list_workspace", {"path": ""}, workspace_root=None)
    assert "не настроен" in result.lower()


@pytest.mark.asyncio
async def test_list_workspace_root(tmp_path: Path) -> None:
    (tmp_path / "diary").mkdir()
    (tmp_path / "file.md").write_text("x", encoding="utf-8")

    result = await execute_tool(
        "list_workspace",
        {"path": ""},
        workspace_root=tmp_path,
    )
    assert "diary/" in result
    assert "file.md" in result


@pytest.mark.asyncio
async def test_list_workspace_subdir(tmp_path: Path) -> None:
    (tmp_path / "diary").mkdir()
    (tmp_path / "diary" / "today.md").write_text("x", encoding="utf-8")

    result = await execute_tool(
        "list_workspace",
        {"path": "diary"},
        workspace_root=tmp_path,
    )
    assert "today.md" in result


@pytest.mark.asyncio
async def test_list_workspace_missing_dir(tmp_path: Path) -> None:
    result = await execute_tool(
        "list_workspace",
        {"path": "nonexistent"},
        workspace_root=tmp_path,
    )
    assert "не найдена" in result.lower()


@pytest.mark.asyncio
async def test_list_workspace_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    result = await execute_tool(
        "list_workspace",
        {"path": "empty"},
        workspace_root=tmp_path,
    )
    assert "пуста" in result.lower()


@pytest.mark.asyncio
async def test_list_workspace_traversal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    result = await execute_tool(
        "list_workspace",
        {"path": ".."},
        workspace_root=ws,
    )
    assert "запрещён" in result.lower()


# --- operator recovery tools ---


@pytest.mark.asyncio
async def test_rollback_morning_personality_tail_requires_operator_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "personality").mkdir()

    blocked = await execute_tool(
        "rollback_morning_personality_tail",
        {"approval_id": "job-c2d0b8d08a084b07808454272612ec5c"},
        workspace_root=tmp_path,
    )
    wrong_approval = await execute_tool(
        "rollback_morning_personality_tail",
        {"approval_id": "wrong"},
        workspace_root=tmp_path,
        context_metadata={"source": "vscode", "source_actor": "codex"},
    )

    assert "только VS Code/Codex" in blocked
    assert "Неверный approval_id" in wrong_approval


@pytest.mark.asyncio
async def test_rollback_morning_personality_tail_moves_and_trims_bad_tail(
    tmp_path: Path,
) -> None:
    personality = tmp_path / "personality"
    (personality / "personal_facts").mkdir(parents=True)
    (personality / "preferences").mkdir()
    (personality / "values").mkdir()
    (personality / "history").mkdir()
    (personality / "personal_facts" / "home_devices.md").write_text(
        "# bad promoted file\n",
        encoding="utf-8",
    )
    (personality / "preferences" / "direct_answers.md").write_text(
        "# direct answers\n\n"
        "<!-- staging review 2026-05-20 10:57 episode 1862 -->\n"
        "bad direct answer tail.\n",
        encoding="utf-8",
    )
    (personality / "personal_facts" / "multi_device_access.md").write_text(
        "# multi device\n\n"
        "<!-- staged 2026-04-22 16:36 episode 1229 -->\n"
        "keep this.\n\n"
        "<!-- staging review 2026-05-21 02:49 episode 1967 -->\n"
        "bad access tail.\n",
        encoding="utf-8",
    )
    (personality / "dreams.md").write_text(
        "# Dreams\n"
        "- [2026-04-06→2026-05-24, merged] bad merged dream\n"
        "- [2026-05-14] keep dream\n",
        encoding="utf-8",
    )
    (personality / "history" / "dreams_archive.md").write_text(
        "# Архив мечт\n\n"
        "keep archive.\n\n"
        "## Объединённые 2026-05-24\n\n"
        "bad archive tail\n",
        encoding="utf-8",
    )
    (personality / "reinforcements.md").write_text("bad generated\n", encoding="utf-8")
    (personality / "MEMORY.md").write_text(
        "- [home devices](personal_facts/home_devices.md) — bad\n"
        "- [episode 1862](insights/episode_1862.md) — bad\n"
        "- [episode 905](insights/episode_905.md) — keep\n",
        encoding="utf-8",
    )
    marker = personality / ".last-consolidated-at"
    marker.write_text("1778795934.000000\n", encoding="utf-8")

    result = await execute_tool(
        "rollback_morning_personality_tail",
        {"approval_id": "job-c2d0b8d08a084b07808454272612ec5c"},
        workspace_root=tmp_path,
        context_metadata={"source": "vscode", "source_actor": "codex"},
    )

    rollback_root = (
        tmp_path
        / "inbox/.rolled_back/2026-05-24T18-39-19-04-personality-tail-gap/personality"
    )
    assert "rollback complete" in result
    assert not (personality / "personal_facts" / "home_devices.md").exists()
    assert (rollback_root / "personal_facts" / "home_devices.md").read_text(
        encoding="utf-8"
    ) == "# bad promoted file\n"
    assert "episode 1862" not in (
        personality / "preferences" / "direct_answers.md"
    ).read_text(encoding="utf-8")
    multi_device = (
        personality / "personal_facts" / "multi_device_access.md"
    ).read_text(encoding="utf-8")
    assert "episode 1229" in multi_device
    assert "episode 1967" not in multi_device
    assert "2026-05-24, merged" not in (personality / "dreams.md").read_text(
        encoding="utf-8"
    )
    assert "Объединённые 2026-05-24" not in (
        personality / "history" / "dreams_archive.md"
    ).read_text(encoding="utf-8")
    assert "Автогенерируется в /morning" in (
        personality / "reinforcements.md"
    ).read_text(encoding="utf-8")
    memory = (personality / "MEMORY.md").read_text(encoding="utf-8")
    assert "home_devices.md" not in memory
    assert "episode_1862.md" not in memory
    assert "episode_905.md" in memory
    assert marker.read_text(encoding="utf-8") == "1778795934.000000\n"


# --- project readonly tools ---


@pytest.mark.asyncio
async def test_read_project_file_success_and_traversal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("VALUE = 'needle'\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    result = await execute_tool(
        "read_project_file",
        {"path": "src/main.py"},
        project_root=project,
    )
    assert "VALUE = 'needle'" in result

    blocked = await execute_tool(
        "read_project_file",
        {"path": "../secret.txt"},
        project_root=project,
    )
    assert "запрещён" in blocked.lower()
    assert "secret" not in blocked


@pytest.mark.asyncio
async def test_read_project_file_line_window(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "main.py"
    target.write_text(
        "\n".join(f"line {index}" for index in range(1, 11)),
        encoding="utf-8",
    )

    result = await execute_tool(
        "read_project_file",
        {"path": "main.py", "start_line": 4, "max_lines": 3},
        project_root=project,
    )

    assert "main.py:4: line 4" in result
    assert "main.py:6: line 6" in result
    assert "main.py:3: line 3" not in result
    assert "main.py:7: line 7" not in result
    assert "всего строк: 10" in result


@pytest.mark.asyncio
async def test_list_project_files_and_search_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / ".git").mkdir()
    (project / "src" / "main.py").write_text("def target_handler(): pass\n")
    (project / ".git" / "config").write_text("target_handler should stay hidden\n")

    listed = await execute_tool(
        "list_project_files",
        {"path": ""},
        project_root=project,
    )
    assert "src/main.py" in listed
    assert ".git/config" not in listed

    found = await execute_tool(
        "search_project",
        {"query": "target_handler"},
        project_root=project,
    )
    assert "src/main.py" in found
    assert ".git/config" not in found


@pytest.mark.asyncio
async def test_project_tools_require_project_root() -> None:
    result = await execute_tool("list_project_files", {"path": ""})
    assert "project root не настроен" in result.lower()


# --- add_knowledge ---


@pytest.mark.asyncio
async def test_add_knowledge_no_store() -> None:
    result = await execute_tool(
        "add_knowledge",
        {"title": "Test", "content": "Body"},
        knowledge_store=None,
    )
    assert "не подключена" in result.lower()


@pytest.mark.asyncio
async def test_add_knowledge_no_title() -> None:
    result = await execute_tool(
        "add_knowledge",
        {"title": "", "content": "Body"},
        knowledge_store=AsyncMock(),
    )
    assert "заголовок" in result.lower()


@pytest.mark.asyncio
async def test_add_knowledge_no_content() -> None:
    result = await execute_tool(
        "add_knowledge",
        {"title": "Title", "content": ""},
        knowledge_store=AsyncMock(),
    )
    assert "содержимое" in result.lower()


@pytest.mark.asyncio
async def test_add_knowledge_success() -> None:
    store = AsyncMock()
    store.add_entry = AsyncMock(return_value=42)

    result = await execute_tool(
        "add_knowledge",
        {
            "title": "My Note",
            "content": "Some text",
            "tags": "python, ai",
            "category": "tools.dev",
        },
        knowledge_store=store,
    )
    assert "42" in result
    assert "My Note" in result
    store.add_entry.assert_awaited_once()


# --- read_workspace_file: empty file ---


@pytest.mark.asyncio
async def test_read_workspace_file_empty_content(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    result = await execute_tool(
        "read_workspace_file",
        {"path": "empty.md"},
        workspace_root=tmp_path,
    )
    assert "пуст" in result.lower()


# --- classify_approval_llm ---


@pytest.mark.asyncio
async def test_classify_approval_llm_yes() -> None:
    """LLM classification returns valid category."""
    from unittest.mock import patch

    from src.llm.protocols import LLMResponse, LLMUsage
    from src.skills.chat_response.skill import _classify_approval_llm

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(text="yes", model="haiku", usage=LLMUsage())
    )

    with patch(
        "src.skills.chat_response.skill.get_router",
        return_value=mock_router,
    ):
        result = await _classify_approval_llm("конечно запиши пожалуйста")

    assert result == "yes"


@pytest.mark.asyncio
async def test_classify_approval_llm_failure_returns_ambiguous() -> None:
    """On LLM failure, returns 'ambiguous' (never raises)."""
    from unittest.mock import patch

    from src.skills.chat_response.skill import _classify_approval_llm

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(side_effect=Exception("API down"))

    with patch(
        "src.skills.chat_response.skill.get_router",
        return_value=mock_router,
    ):
        result = await _classify_approval_llm("something")

    assert result == "ambiguous"


@pytest.mark.asyncio
async def test_classify_approval_llm_garbage_returns_ambiguous() -> None:
    """If LLM returns invalid text, returns 'ambiguous'."""
    from unittest.mock import patch

    from src.llm.protocols import LLMResponse, LLMUsage
    from src.skills.chat_response.skill import _classify_approval_llm

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(
            text="definitely maybe", model="haiku", usage=LLMUsage()
        )
    )

    with patch(
        "src.skills.chat_response.skill.get_router",
        return_value=mock_router,
    ):
        result = await _classify_approval_llm("test")

    assert result == "ambiguous"


# --- classify_approval (two-tier) ---


@pytest.mark.asyncio
async def test_classify_approval_fast_path() -> None:
    """Known pattern bypasses LLM entirely."""
    from src.skills.chat_response.skill import classify_approval

    result = await classify_approval("да")
    assert result == "yes"


@pytest.mark.asyncio
async def test_classify_approval_llm_fallback() -> None:
    """Ambiguous input triggers LLM classification."""
    from unittest.mock import patch

    from src.llm.protocols import LLMResponse, LLMUsage
    from src.skills.chat_response.skill import classify_approval

    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(
        return_value=LLMResponse(text="no", model="haiku", usage=LLMUsage())
    )

    with patch(
        "src.skills.chat_response.skill.get_router",
        return_value=mock_router,
    ):
        result = await classify_approval("расскажи подробнее")

    assert result == "no"
    mock_router.generate.assert_awaited_once()


# --- post_to_channel ---


def _make_mock_bot() -> AsyncMock:
    """Create a mock aiogram Bot that returns a message with message_id."""
    bot = AsyncMock()
    msg = MagicMock()
    msg.message_id = 42
    bot.send_message = AsyncMock(return_value=msg)
    return bot


@pytest.mark.asyncio
async def test_post_to_channel_success(tmp_path: Path) -> None:
    """Sends message via bot and archives to workspace."""
    bot = _make_mock_bot()
    result = await execute_tool(
        "post_to_channel",
        {"text": "Тестовый пост #AI"},
        bot=bot,
        channel_id="-1001234567890",
        workspace_root=tmp_path,
    )
    assert "Опубликовано" in result
    bot.send_message.assert_awaited_once_with(
        chat_id="-1001234567890",
        text="Тестовый пост #AI",
        parse_mode=None,
    )
    # Archive created
    posts = list((tmp_path / "channel" / "posts").glob("*.md"))
    assert len(posts) == 1
    content = posts[0].read_text()
    assert "message_id: 42" in content
    assert "Тестовый пост #AI" in content


@pytest.mark.asyncio
async def test_post_to_channel_splits_long_text(tmp_path: Path) -> None:
    """Long post_to_channel calls publish all chunks and archive original text."""
    bot = _make_mock_bot()
    bot.send_message.side_effect = [
        MagicMock(message_id=message_id) for message_id in range(301, 311)
    ]
    long_text = ("длинный пост\n\n" * 700).strip()

    result = await execute_tool(
        "post_to_channel",
        {"text": long_text},
        bot=bot,
        channel_id="-1001234567890",
        workspace_root=tmp_path,
    )

    assert "Опубликовано" in result
    assert bot.send_message.await_count > 1
    for call in bot.send_message.await_args_list:
        assert call.kwargs["chat_id"] == "-1001234567890"
        assert len(call.kwargs["text"]) <= 4096
    posts = list((tmp_path / "channel" / "posts").glob("*.md"))
    assert len(posts) == 1
    content = posts[0].read_text()
    assert "message_id: 301" in content
    assert long_text in content


@pytest.mark.asyncio
async def test_post_to_channel_publishes_full_workspace_file(tmp_path: Path) -> None:
    """Publishing by path reads the full outbox file and archives the draft."""
    bot = _make_mock_bot()
    bot.send_message.side_effect = [
        MagicMock(message_id=401),
        MagicMock(message_id=402),
    ]
    draft_dir = tmp_path / "outbox" / "channel_posts"
    draft_dir.mkdir(parents=True)
    draft_path = draft_dir / "2026-05-06.md"
    long_text = ("полный файл поста\n\n" * 400).strip()
    draft_path.write_text(long_text, encoding="utf-8")

    result = await execute_tool(
        "post_to_channel",
        {"path": "outbox/channel_posts/2026-05-06.md"},
        bot=bot,
        channel_id="-1001234567890",
        workspace_root=tmp_path,
    )

    assert "Опубликовано" in result
    assert bot.send_message.await_count > 1
    sent_text = "".join(
        call.kwargs["text"] for call in bot.send_message.await_args_list
    )
    assert sent_text.replace("\n", "") == long_text.replace("\n", "")
    posts = list((tmp_path / "channel" / "posts").glob("*.md"))
    assert len(posts) == 1
    archive = posts[0].read_text()
    assert "message_id: 401" in archive
    assert long_text in archive
    assert not draft_path.exists()
    assert (tmp_path / "outbox" / ".processed" / "2026-05-06.md").exists()


@pytest.mark.asyncio
async def test_post_to_channel_empty_text() -> None:
    result = await execute_tool(
        "post_to_channel",
        {"text": ""},
        bot=_make_mock_bot(),
        channel_id="-100123",
    )
    assert "пустой" in result.lower()


@pytest.mark.asyncio
async def test_post_to_channel_whitespace_only() -> None:
    result = await execute_tool(
        "post_to_channel",
        {"text": "   \n  "},
        bot=_make_mock_bot(),
        channel_id="-100123",
    )
    assert "пустой" in result.lower()


@pytest.mark.asyncio
async def test_post_to_channel_no_bot() -> None:
    result = await execute_tool(
        "post_to_channel",
        {"text": "test"},
        bot=None,
        channel_id="-100123",
    )
    assert "бот" in result.lower() or "bot" in result.lower()


@pytest.mark.asyncio
async def test_post_to_channel_no_channel_id() -> None:
    result = await execute_tool(
        "post_to_channel",
        {"text": "test"},
        bot=_make_mock_bot(),
        channel_id="",
    )
    assert "канал" in result.lower()


@pytest.mark.asyncio
async def test_post_to_channel_telegram_error() -> None:
    bot = _make_mock_bot()
    bot.send_message.side_effect = Exception("Forbidden: bot blocked")
    result = await execute_tool(
        "post_to_channel",
        {"text": "test"},
        bot=bot,
        channel_id="-100123",
    )
    assert "ошибка" in result.lower()
