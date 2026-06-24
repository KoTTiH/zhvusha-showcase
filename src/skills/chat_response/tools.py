"""Chat tools for agentic response loop.

Defines tools that Zhvusha can invoke during chat to verify claims,
search knowledge, and read workspace files. Each tool is a simple
async function that returns a string result.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.file_access import safe_path
from src.llm.protocols import ToolDefinition
from src.utils.telegram import send_long_message

if TYPE_CHECKING:
    from aiogram import Bot

    from src.knowledge import KnowledgeStore

logger = structlog.get_logger()

# Max 5 tool calls per response (anti-loop safety)
MAX_TOOL_CALLS = 5
_KNOWLEDGE_TOOL_NAMES = frozenset(
    {"search_knowledge", "get_entry_content", "browse_categories"}
)
_READ_TOOL_NAMES = frozenset(
    {
        "read_workspace_file",
        "list_workspace",
        "read_project_file",
        "list_project_files",
        "search_project",
    }
)
_PROJECT_IGNORE_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_PROJECT_MAX_FILE_CHARS = 6000
_PROJECT_DEFAULT_LINE_WINDOW = 120
_PROJECT_MAX_LINE_WINDOW = 240
_PROJECT_MAX_LISTED_FILES = 100
_PROJECT_MAX_SEARCH_FILES = 1000
_PROJECT_MAX_SEARCH_RESULTS = 50
_PROJECT_MAX_SEARCH_FILE_BYTES = 512 * 1024
_MORNING_RECOVERY_APPROVAL_ID = "job-c2d0b8d08a084b07808454272612ec5c"
_MORNING_RECOVERY_ROLLBACK_ROOT = (
    "inbox/.rolled_back/2026-05-24T18-39-19-04-personality-tail-gap/personality"
)
_MORNING_RECOVERY_MOVE_FILES = (
    "personal_facts/home_devices.md",
    "preferences/physical_ai_projects.md",
    "skills/ai_cto_project_review.md",
    "preferences/browser_verification.md",
)
_MORNING_RECOVERY_INDEX_PATHS = (
    "personal_facts/home_devices.md",
    "preferences/physical_ai_projects.md",
    "skills/ai_cto_project_review.md",
    "preferences/browser_verification.md",
)
_MORNING_RECOVERY_BAD_EPISODE_IDS = (
    "1788",
    "1790",
    "1862",
    "1870",
    "1894",
    "1908",
    "1912",
    "1932",
    "1934",
    "1947",
    "1949",
    "1951",
    "1955",
    "1997",
    "1999",
    "2001",
    "2003",
    "2005",
    "2007",
    "2009",
    "2037",
    "2043",
    "2046",
    "2047",
    "2051",
    "2073",
    "2103",
    "2109",
    "2111",
    "2115",
    "2117",
    "2121",
    "2127",
    "2129",
    "2131",
)
_MORNING_RECOVERY_MARKER_PATTERNS = {
    "preferences/direct_answers.md": ("episode 1862",),
    "values/autonomy_through_gates.md": (
        "episode 1955",
        "episode 2001",
        "episode 2115",
    ),
    "values/verify_before_conclude.md": (
        "episode 1788",
        "episode 1790",
        "episode 1997",
        "episode 1999",
        "episode 2003",
    ),
    "personal_facts/multi_device_access.md": ("episode 1967",),
    "wishlist.md": ("episode 2109",),
}
_REINFORCEMENTS_MINIMAL = (
    "# Reinforcement Patterns\n\n"
    "Автогенерируется в /morning из feedback-эпизодов. Не редактировать вручную.\n"
)


def get_chat_tools(
    context_metadata: dict[str, Any] | None = None,
) -> list[ToolDefinition]:
    """Return tool definitions for the chat agentic loop."""
    tools = [
        ToolDefinition(
            name="search_knowledge",
            description=(
                "Поиск по базе знаний Жвуши (PostgreSQL + pgvector). "
                "Используй когда нужно проверить факт, найти информацию из каналов, "
                "YouTube, браузера или других внешних источников."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    },
                },
                "required": ["query"],
            },
        ),
        ToolDefinition(
            name="get_entry_content",
            description=(
                "Получить полный текст записи из базы знаний по ID. "
                "Используй после search_knowledge, когда нужно прочитать "
                "конкретную запись целиком."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "integer",
                        "description": "ID записи из результатов поиска",
                    },
                },
                "required": ["entry_id"],
            },
        ),
        ToolDefinition(
            name="browse_categories",
            description=(
                "Показать дерево категорий базы знаний с количеством записей. "
                "Используй чтобы узнать структуру знаний и найти нужную область."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
        ),
        ToolDefinition(
            name="read_workspace_file",
            description=(
                "Прочитать файл из workspace (дневник, personality, заметки). "
                "Используй когда нужно перечитать то, что ты писала раньше."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Относительный путь от workspace root, "
                            "например: diary/2026-04-08.md"
                        ),
                    },
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="list_workspace",
            description=(
                "Показать содержимое директории workspace. "
                "Используй чтобы узнать какие файлы и папки существуют."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Относительный путь от workspace root, "
                            "например: diary/ или personality/. "
                            "Пустая строка — корень workspace."
                        ),
                        "default": "",
                    },
                },
            },
        ),
        ToolDefinition(
            name="read_project_file",
            description=(
                "Прочитать файл из текущего VS Code project root. "
                "Используй для read-only проверки кода, тестов и документации "
                "проекта. Не читает файлы за пределами project root."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь от project root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": (
                            "Опциональная первая строка для чтения большого файла."
                        ),
                        "default": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "Опциональное число строк, максимум "
                            f"{_PROJECT_MAX_LINE_WINDOW}."
                        ),
                        "default": _PROJECT_DEFAULT_LINE_WINDOW,
                    },
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="list_project_files",
            description=(
                "Показать файлы в текущем VS Code project root без служебных "
                "директорий вроде .git, .venv и node_modules."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь от project root.",
                        "default": "",
                    },
                },
            },
        ),
        ToolDefinition(
            name="search_project",
            description=(
                "Read-only поиск строки по текстовым файлам текущего VS Code "
                "project root. Используй чтобы найти символы, обработчики, "
                "тесты и документы перед ответом о коде."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Строка для поиска.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Опциональный относительный подкаталог.",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        ),
        ToolDefinition(
            name="add_knowledge",
            description=(
                "Добавить новую запись в базу знаний. "
                "Используй когда Никита просит запомнить факт, решение, "
                "или ты сама хочешь сохранить важную информацию."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Заголовок записи",
                    },
                    "content": {
                        "type": "string",
                        "description": "Содержимое записи (markdown)",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Теги через запятую",
                        "default": "",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Путь категории, например: tools.python или dev.roadmap"
                        ),
                        "default": "",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        ToolDefinition(
            name="post_to_channel",
            description=(
                "Опубликовать сообщение в Telegram-канал @zhvusha. "
                "Используй когда Никита просит выложить пост или "
                "когда контент готов к публикации."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Текст поста для публикации в канал. Можно оставить "
                            "пустым, если указан path."
                        ),
                        "default": "",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Относительный путь к готовому посту в workspace, "
                            "например outbox/channel_posts/2026-05-06.md. "
                            "Используй этот вариант для готовых outbox-постов, "
                            "чтобы опубликовать полный файл без обрезки."
                        ),
                        "default": "",
                    },
                },
            },
        ),
    ]
    if _computer_use_tool_enabled(context_metadata):
        tools.append(
            ToolDefinition(
                name="computer_use",
                description=(
                    "Выполнить одно scoped computer/browser действие через "
                    "central computer-use skill gate. Используй для живого "
                    "браузера, кликов, прокрутки, desktop screenshot/app/hotkeys "
                    "и продолжения BODY_OBSERVATION computer-use workflow. "
                    "Передавай структурированный payload, не текстовую "
                    "/computer_use команду."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": (
                                "computer-use action, например browser_status, "
                                "browser_navigate, browser_click, "
                                "browser_interactive_task, browser_scroll, "
                                "desktop_screenshot"
                            ),
                        },
                        "url": {"type": "string", "default": ""},
                        "target": {"type": "string", "default": ""},
                        "selector": {"type": "string", "default": ""},
                        "text": {"type": "string", "default": ""},
                        "goal": {"type": "string", "default": ""},
                        "operation": {"type": "string", "default": ""},
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "artifact_requirements": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "default": {},
                        },
                        "success_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "default": {},
                        },
                    },
                    "required": ["action"],
                },
            )
        )
    if _operator_recovery_tools_enabled(context_metadata):
        tools.append(
            ToolDefinition(
                name="rollback_morning_personality_tail",
                description=(
                    "Операторский recovery-tool для отката хвоста плохой "
                    "/morning-консолидации 2026-05-24. Доступен только из "
                    "VS Code/Codex контекста и требует точный approval_id."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "approval_id": {
                            "type": "string",
                            "description": "Approval id для текущего recovery.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Только показать план без записи.",
                            "default": False,
                        },
                    },
                    "required": ["approval_id"],
                },
            )
        )
    return tools


def _computer_use_tool_enabled(context_metadata: dict[str, Any] | None) -> bool:
    if not isinstance(context_metadata, dict):
        return False
    return bool(context_metadata.get("computer_use_tool_enabled") is True)


async def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    knowledge_store: KnowledgeStore | None = None,
    workspace_root: Path | None = None,
    project_root: Path | str | None = None,
    bot: Bot | None = None,
    channel_id: str = "",
    context_metadata: dict[str, Any] | None = None,
) -> str:
    """Execute a chat tool and return result string.

    Never raises — returns error message on failure.
    """
    try:
        if tool_name in _KNOWLEDGE_TOOL_NAMES:
            return await _execute_knowledge_tool(
                tool_name,
                tool_input,
                knowledge_store=knowledge_store,
            )
        if tool_name in _READ_TOOL_NAMES:
            return _execute_read_tool(
                tool_name,
                tool_input,
                workspace_root=workspace_root,
                project_root=project_root,
            )
        if tool_name == "add_knowledge":
            return await _add_knowledge(
                title=tool_input.get("title", ""),
                content=tool_input.get("content", ""),
                tags=tool_input.get("tags", ""),
                category=tool_input.get("category", ""),
                knowledge_store=knowledge_store,
            )
        if tool_name == "post_to_channel":
            return await _post_to_channel(
                text=tool_input.get("text", ""),
                path=tool_input.get("path", ""),
                bot=bot,
                channel_id=channel_id,
                workspace_root=workspace_root,
            )
        if tool_name == "rollback_morning_personality_tail":
            return _rollback_morning_personality_tail(
                approval_id=str(tool_input.get("approval_id", "")),
                dry_run=bool(tool_input.get("dry_run", False)),
                workspace_root=workspace_root,
                context_metadata=context_metadata,
            )
        return f"Неизвестный инструмент: {tool_name}"
    except Exception:
        logger.warning("chat_tool_failed", tool=tool_name, exc_info=True)
        return f"Ошибка при выполнении {tool_name}"


async def _execute_knowledge_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    knowledge_store: KnowledgeStore | None,
) -> str:
    if tool_name == "search_knowledge":
        return await _search_knowledge(
            tool_input.get("query", ""),
            knowledge_store=knowledge_store,
        )
    if tool_name == "get_entry_content":
        return await _get_entry_content(
            tool_input.get("entry_id", 0),
            knowledge_store=knowledge_store,
        )
    return await _browse_categories(knowledge_store=knowledge_store)


def _execute_read_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    workspace_root: Path | None,
    project_root: Path | str | None,
) -> str:
    if tool_name == "read_workspace_file":
        return _read_workspace_file(
            tool_input.get("path", ""),
            workspace_root=workspace_root,
        )
    if tool_name == "list_workspace":
        return _list_workspace(
            tool_input.get("path", ""),
            workspace_root=workspace_root,
        )
    if tool_name == "read_project_file":
        return _read_project_file(
            tool_input.get("path", ""),
            start_line=tool_input.get("start_line"),
            max_lines=tool_input.get("max_lines"),
            project_root=project_root,
        )
    if tool_name == "list_project_files":
        return _list_project_files(
            tool_input.get("path", ""),
            project_root=project_root,
        )
    return _search_project(
        query=tool_input.get("query", ""),
        path=tool_input.get("path", ""),
        project_root=project_root,
    )


async def _search_knowledge(
    query: str,
    *,
    knowledge_store: KnowledgeStore | None = None,
) -> str:
    """Search knowledge base and return formatted results."""
    if not query.strip():
        return "Пустой запрос."
    if knowledge_store is None:
        return "Knowledge base не подключена."

    # limit=5 (vs 3 in single-shot path) — agentic loop benefits from
    # more context since LLM explicitly chose to search.
    results = await knowledge_store.hybrid_search(query, limit=5)
    if not results:
        return f"Ничего не найдено по запросу «{query}»."

    ids = [r.id for r in results if r.id is not None]
    summaries = await knowledge_store.get_summaries(ids)

    parts: list[str] = []
    for s in summaries:
        parts.append(f"• [#{s.id}] {s.title}: {s.summary or '(без описания)'}")

    return "\n".join(parts)


async def _get_entry_content(
    entry_id: int,
    *,
    knowledge_store: KnowledgeStore | None = None,
) -> str:
    """Get full content of a knowledge base entry by ID."""
    if not entry_id:
        return "Не указан ID записи."
    if knowledge_store is None:
        return "Knowledge base не подключена."

    entry = await knowledge_store.get_full_content(entry_id)
    if entry is None:
        return f"Запись #{entry_id} не найдена."

    parts = [f"# {entry.title}"]
    if entry.tags:
        parts.append(f"Теги: {', '.join(entry.tags)}")
    if entry.category_name_ru:
        parts.append(f"Категория: {entry.category_name_ru}")
    parts.append("")
    parts.append(entry.content)

    result = "\n".join(parts)
    max_chars = 6000
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (обрезано)"
    return result


async def _browse_categories(
    *,
    knowledge_store: KnowledgeStore | None = None,
) -> str:
    """Browse knowledge base category tree."""
    if knowledge_store is None:
        return "Knowledge base не подключена."

    categories = await knowledge_store.browse_categories()
    if not categories:
        return "Категорий пока нет."

    lines: list[str] = []
    for cat in categories:
        lines.append(f"• {cat.name_ru} ({cat.path}) — {cat.entry_count} записей")

    return "\n".join(lines)


def _read_workspace_file(
    path: str,
    *,
    workspace_root: Path | None = None,
) -> str:
    """Read a workspace file by relative path."""
    if not path.strip():
        return "Пустой путь."
    if workspace_root is None:
        return "Workspace не настроен."

    # Security: prevent path traversal (reuses safe_path with symlink check)
    full_path = safe_path(workspace_root, path)
    if full_path is None:
        return "Доступ запрещён: путь за пределами workspace."

    if not full_path.is_file():
        return f"Файл не найден: {path}"

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"Бинарный workspace-файл найден: {path}. "
            "Его не нужно читать как текст; если это image artifact, укажи "
            "этот путь в ответе, и слой доставки прикрепит файл."
        )
    except OSError:
        return f"Ошибка чтения: {path}"

    if not content:
        return f"Файл пуст: {path}"

    # Truncate large files
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (обрезано)"

    return content


def _list_workspace(
    path: str,
    *,
    workspace_root: Path | None = None,
) -> str:
    """List files and directories in workspace."""
    if workspace_root is None:
        return "Workspace не настроен."

    target = safe_path(workspace_root, path) if path.strip() else workspace_root
    if target is None:
        return "Доступ запрещён: путь за пределами workspace."

    if not target.is_dir():
        return f"Директория не найдена: {path}"

    max_entries = 50
    entries: list[str] = []
    try:
        for item in sorted(target.iterdir()):
            rel = item.relative_to(workspace_root)
            suffix = "/" if item.is_dir() else ""
            entries.append(f"  {rel}{suffix}")
            if len(entries) >= max_entries:
                entries.append(f"  ... (ещё файлы, показано {max_entries})")
                break
    except OSError:
        return f"Ошибка чтения директории: {path}"

    if not entries:
        return f"Директория пуста: {path or '/'}"

    return "\n".join(entries)


def _operator_recovery_tools_enabled(
    metadata: dict[str, Any] | None,
) -> bool:
    return (
        isinstance(metadata, dict)
        and str(metadata.get("source")) == "vscode"
        and str(metadata.get("source_actor")) == "codex"
    )


def _rollback_morning_personality_tail(
    *,
    approval_id: str,
    dry_run: bool,
    workspace_root: Path | None,
    context_metadata: dict[str, Any] | None,
) -> str:
    """Rollback the approved 2026-05-24 morning recovery personality tail."""
    validation_error = _validate_morning_recovery_request(
        approval_id=approval_id,
        context_metadata=context_metadata,
    )
    if validation_error is not None:
        return validation_error
    paths = _morning_recovery_paths(workspace_root)
    if isinstance(paths, str):
        return paths

    recorder = _RollbackRecorder(
        personality_dir=paths[0],
        rollback_dir=paths[1],
        dry_run=dry_run,
    )
    _move_morning_recovery_files(recorder)
    _trim_morning_recovery_marker_files(recorder)
    _trim_morning_recovery_dreams(recorder)
    _reset_morning_recovery_reinforcements(recorder)
    _trim_morning_recovery_memory(recorder)
    return recorder.render()


class _RollbackRecorder:
    def __init__(
        self,
        *,
        personality_dir: Path,
        rollback_dir: Path,
        dry_run: bool,
    ) -> None:
        self.personality_dir = personality_dir
        self.rollback_dir = rollback_dir
        self.dry_run = dry_run
        self.planned: list[str] = []
        self.changed: list[str] = []

    def backup(self, rel: str) -> None:
        source = self.personality_dir / rel
        if not source.exists():
            return
        target = self.rollback_dir / f"{rel}.before_tail_gap_rollback"
        self.planned.append(f"backup {rel}")
        if self.dry_run:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def move(self, rel: str) -> None:
        source = self.personality_dir / rel
        if not source.exists():
            return
        target = self.rollback_dir / rel
        self.planned.append(f"move {rel}")
        if self.dry_run:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self.changed.append(rel)

    def write_cleaned(self, rel: str, cleaned: str, *, action: str = "trim") -> None:
        self.backup(rel)
        self.planned.append(f"{action} {rel}")
        if self.dry_run:
            return
        (self.personality_dir / rel).write_text(cleaned, encoding="utf-8")
        self.changed.append(rel)

    def render(self) -> str:
        if self.dry_run:
            return "dry-run morning rollback:\n" + "\n".join(
                self.planned or ["nothing"]
            )
        return "morning personality tail rollback complete; changed: " + (
            ", ".join(self.changed) if self.changed else "nothing"
        )


def _validate_morning_recovery_request(
    *,
    approval_id: str,
    context_metadata: dict[str, Any] | None,
) -> str | None:
    if not _operator_recovery_tools_enabled(context_metadata):
        return (
            "rollback_morning_personality_tail доступен только VS Code/Codex operator."
        )
    if approval_id.strip() != _MORNING_RECOVERY_APPROVAL_ID:
        return "Неверный approval_id для morning rollback."
    return None


def _morning_recovery_paths(workspace_root: Path | None) -> tuple[Path, Path] | str:
    if workspace_root is None:
        return "Workspace не настроен."

    personality_dir = safe_path(workspace_root, "personality")
    rollback_dir = safe_path(workspace_root, _MORNING_RECOVERY_ROLLBACK_ROOT)
    if personality_dir is None or rollback_dir is None:
        return "Доступ запрещён: recovery path вне workspace."
    if not personality_dir.is_dir():
        return "personality/ не найден."
    return personality_dir, rollback_dir


def _move_morning_recovery_files(recorder: _RollbackRecorder) -> None:
    for rel in _MORNING_RECOVERY_MOVE_FILES:
        recorder.move(rel)


def _trim_morning_recovery_marker_files(recorder: _RollbackRecorder) -> None:
    for rel, markers in _MORNING_RECOVERY_MARKER_PATTERNS.items():
        path = recorder.personality_dir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        cleaned = _remove_staging_review_blocks(text, markers)
        if cleaned != text:
            recorder.write_cleaned(rel, cleaned)


def _trim_morning_recovery_dreams(recorder: _RollbackRecorder) -> None:
    _trim_morning_recovery_dreams_current(recorder)
    _trim_morning_recovery_dreams_archive(recorder)


def _trim_morning_recovery_dreams_current(recorder: _RollbackRecorder) -> None:
    rel = "dreams.md"
    path = recorder.personality_dir / rel
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    cleaned = re.sub(
        r"\n- \[2026-04-06→2026-05-24, merged\].*(?=\n|$)",
        "",
        text,
    )
    if cleaned != text:
        recorder.write_cleaned(rel, cleaned)


def _trim_morning_recovery_dreams_archive(recorder: _RollbackRecorder) -> None:
    rel = "history/dreams_archive.md"
    path = recorder.personality_dir / rel
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    cleaned = re.sub(r"\n## Объединённые 2026-05-24\n.*\Z", "\n", text, flags=re.S)
    if cleaned != text:
        recorder.write_cleaned(rel, cleaned.rstrip() + "\n")


def _reset_morning_recovery_reinforcements(recorder: _RollbackRecorder) -> None:
    rel = "reinforcements.md"
    if (recorder.personality_dir / rel).exists():
        recorder.write_cleaned(rel, _REINFORCEMENTS_MINIMAL, action="reset")


def _trim_morning_recovery_memory(recorder: _RollbackRecorder) -> None:
    rel = "MEMORY.md"
    path = recorder.personality_dir / rel
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    cleaned = _remove_memory_index_tail(text)
    if cleaned != text:
        recorder.write_cleaned(rel, cleaned)


def _remove_staging_review_blocks(text: str, markers: tuple[str, ...]) -> str:
    cleaned = text
    for marker in markers:
        cleaned = re.sub(
            rf"\n?<!-- staging review [^>]*{re.escape(marker)} -->\n.*?"
            r"(?=\n\n<!--|\n\n#|\Z)",
            "",
            cleaned,
            flags=re.S,
        )
    return _normalize_blank_lines(cleaned)


def _remove_memory_index_tail(text: str) -> str:
    bad_episode_refs = tuple(
        f"episode_{episode_id}.md" for episode_id in _MORNING_RECOVERY_BAD_EPISODE_IDS
    )
    kept: list[str] = []
    for line in text.splitlines():
        if any(path in line for path in _MORNING_RECOVERY_INDEX_PATHS):
            continue
        if any(ref in line for ref in bad_episode_refs):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def _normalize_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


def _project_root(project_root: Path | str | None) -> Path | None:
    if project_root is None:
        return None
    try:
        root = Path(project_root).expanduser().resolve()
    except OSError:
        return None
    return root if root.is_dir() else None


def _read_project_file(
    path: str,
    *,
    start_line: object = None,
    max_lines: object = None,
    project_root: Path | str | None = None,
) -> str:
    content, error = _read_project_file_content(path, project_root=project_root)
    if error is not None:
        return error
    if content is None:
        return f"Файл не найден: {path}"
    if not content:
        return f"Файл пуст: {path}"
    if start_line is not None or max_lines is not None:
        return _format_project_line_window(
            path=path,
            content=content,
            start_line=start_line,
            max_lines=max_lines,
        )
    if len(content) > _PROJECT_MAX_FILE_CHARS:
        content = content[:_PROJECT_MAX_FILE_CHARS] + "\n... (обрезано)"
    return content


def _read_project_file_content(
    path: str,
    *,
    project_root: Path | str | None = None,
) -> tuple[str | None, str | None]:
    if not path.strip():
        return None, "Пустой путь."
    root = _project_root(project_root)
    if root is None:
        return None, "Project root не настроен."
    full_path = safe_path(root, path)
    if full_path is None:
        return None, "Доступ запрещён: путь за пределами project root."
    if not full_path.is_file():
        return None, f"Файл не найден: {path}"
    if _is_ignored_project_path(full_path, root):
        return None, "Доступ запрещён: служебный путь проекта."
    try:
        return full_path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, f"Файл не текстовый: {path}"
    except OSError:
        return None, f"Ошибка чтения: {path}"


def _format_project_line_window(
    *,
    path: str,
    content: str,
    start_line: object,
    max_lines: object,
) -> str:
    lines = content.splitlines()
    start = _positive_int(start_line, default=1)
    count = min(
        _positive_int(max_lines, default=_PROJECT_DEFAULT_LINE_WINDOW),
        _PROJECT_MAX_LINE_WINDOW,
    )
    if start > len(lines):
        return f"Строка вне файла: {path} (всего строк: {len(lines)})"
    end = min(start + count - 1, len(lines))
    selected = lines[start - 1 : end]
    rendered = [f"{path}:{index}: {line}" for index, line in enumerate(selected, start)]
    if end < len(lines):
        rendered.append(f"... (обрезано, всего строк: {len(lines)})")
    return "\n".join(rendered)


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _list_project_files(
    path: str,
    *,
    project_root: Path | str | None = None,
) -> str:
    root = _project_root(project_root)
    if root is None:
        return "Project root не настроен."
    target = safe_path(root, path) if path.strip() else root
    if target is None:
        return "Доступ запрещён: путь за пределами project root."
    if not target.exists():
        return f"Путь не найден: {path}"
    files = [target] if target.is_file() else _iter_project_files(target, root)
    entries: list[str] = []
    for file_path in files:
        if _is_ignored_project_path(file_path, root):
            continue
        with contextlib.suppress(ValueError):
            entries.append(str(file_path.relative_to(root)))
        if len(entries) >= _PROJECT_MAX_LISTED_FILES:
            entries.append(f"... (ещё файлы, показано {_PROJECT_MAX_LISTED_FILES})")
            break
    return "\n".join(entries) if entries else f"Файлы не найдены: {path or '/'}"


def _search_project(
    *,
    query: object,
    path: object = "",
    project_root: Path | str | None = None,
) -> str:
    needle = str(query or "").strip()
    if not needle:
        return "Пустой запрос."
    root = _project_root(project_root)
    if root is None:
        return "Project root не настроен."
    target, target_error = _project_target(root, path)
    if target_error is not None:
        return target_error
    if target is None:
        return "Путь не найден."
    files = [target] if target.is_file() else _iter_project_files(target, root)
    results, scanned = _collect_project_search_results(
        files=files,
        root=root,
        needle_lower=needle.lower(),
    )
    return _format_project_search_results(
        needle=needle,
        results=results,
        scanned=scanned,
    )


def _project_target(root: Path, path: object) -> tuple[Path | None, str | None]:
    raw_path = str(path or "")
    target = safe_path(root, raw_path) if raw_path.strip() else root
    if target is None:
        return None, "Доступ запрещён: путь за пределами project root."
    if not target.exists():
        return None, f"Путь не найден: {raw_path}"
    return target, None


def _collect_project_search_results(
    *,
    files: list[Path],
    root: Path,
    needle_lower: str,
) -> tuple[list[str], int]:
    results: list[str] = []
    scanned = 0
    for file_path in files:
        if _is_ignored_project_path(file_path, root):
            continue
        scanned += 1
        if scanned > _PROJECT_MAX_SEARCH_FILES:
            break
        if not _is_searchable_text_file(file_path):
            continue
        match = _search_file(file_path, root, needle_lower)
        if match:
            results.append(match)
        if len(results) >= _PROJECT_MAX_SEARCH_RESULTS:
            break
    return results, scanned


def _format_project_search_results(
    *,
    needle: str,
    results: list[str],
    scanned: int,
) -> str:
    if not results:
        return f"Ничего не найдено по запросу «{needle}»."
    if scanned > _PROJECT_MAX_SEARCH_FILES:
        results.append(
            f"... (поиск остановлен после {_PROJECT_MAX_SEARCH_FILES} файлов)"
        )
    return "\n".join(results)


def _iter_project_files(target: Path, root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        iterator = target.rglob("*")
        for item in iterator:
            if item.is_file() and not _is_ignored_project_path(item, root):
                files.append(item)
    except OSError:
        return files
    return sorted(files)


def _is_ignored_project_path(path: Path, root: Path) -> bool:
    with contextlib.suppress(ValueError):
        rel = path.relative_to(root)
        return any(part in _PROJECT_IGNORE_PARTS for part in rel.parts)
    return True


def _is_searchable_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > _PROJECT_MAX_SEARCH_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def _search_file(path: Path, root: Path, needle_lower: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    for index, line in enumerate(lines, start=1):
        if needle_lower in line.lower():
            rel = path.relative_to(root)
            preview = line.strip()
            if len(preview) > 160:
                preview = preview[:157] + "..."
            return f"{rel}:{index}: {preview}"
    return ""


async def _add_knowledge(
    *,
    title: str,
    content: str,
    tags: str = "",
    category: str = "",
    knowledge_store: KnowledgeStore | None = None,
) -> str:
    """Add a new entry to the knowledge base."""
    if not title.strip():
        return "Не указан заголовок."
    if not content.strip():
        return "Не указано содержимое."
    if knowledge_store is None:
        return "Knowledge base не подключена."

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    entry_id = await knowledge_store.add_entry(
        title=title.strip(),
        content=content.strip(),
        tags=tag_list,
        category_path=category.strip() or None,
        source="chat_tool",
    )

    return f"Записано в базу знаний: #{entry_id} «{title.strip()}»"


async def _post_to_channel(
    *,
    text: str,
    path: str = "",
    bot: Bot | None = None,
    channel_id: str = "",
    workspace_root: Path | None = None,
) -> str:
    """Post a message to the Telegram channel and archive it."""
    text, draft_path, load_error = _load_channel_post_text(
        text=text,
        path=path,
        workspace_root=workspace_root,
    )
    if load_error is not None:
        return load_error

    text = text.strip()
    if not text:
        return "Пустой текст. Напиши что опубликовать."
    if bot is None:
        return "Бот не доступен."
    if not channel_id:
        return "Канал не настроен."

    messages = await send_long_message(bot, chat_id=channel_id, text=text)
    first_message = messages[0]

    if workspace_root is not None:
        from src.skills.channel_writer.archive import save_published_post

        await save_published_post(
            workspace_root=workspace_root,
            text=text,
            message_id=first_message.message_id,
        )
        if draft_path is not None and _is_channel_outbox_post(
            draft_path,
            workspace_root,
        ):
            processed_dir = workspace_root / "outbox" / ".processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(draft_path), str(processed_dir / draft_path.name))

    logger.info(
        "channel_post_via_tool",
        channel=channel_id,
        message_id=first_message.message_id,
        length=len(text),
        parts=len(messages),
    )

    return f"Опубликовано в канал (message_id: {first_message.message_id})"


def _load_channel_post_text(
    *,
    text: str,
    path: str,
    workspace_root: Path | None,
) -> tuple[str, Path | None, str | None]:
    path = path.strip()
    if not path:
        return text, None, None
    if workspace_root is None:
        return "", None, "Workspace не настроен."

    draft_path = safe_path(workspace_root, path)
    if draft_path is None:
        return "", None, "Доступ запрещён: путь за пределами workspace."
    if not draft_path.is_file():
        return "", None, f"Файл не найден: {path}"
    try:
        return draft_path.read_text(encoding="utf-8"), draft_path, None
    except OSError:
        return "", None, f"Ошибка чтения: {path}"


def _is_channel_outbox_post(path: Path, workspace_root: Path) -> bool:
    try:
        rel = path.relative_to(workspace_root)
    except ValueError:
        return False
    return len(rel.parts) == 3 and rel.parts[:2] == ("outbox", "channel_posts")
