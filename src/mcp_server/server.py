"""MCP server exposing knowledge base tools + web dashboard.

Transports:
- stdio: for local agent clients (spawned per session via .mcp.json)
- sse: for long-running remote/client access via --http flag

Dashboard:
- /dashboard — Obsidian-like knowledge graph UI
- /api/* — REST endpoints for the dashboard
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mcp.server.fastmcp import FastMCP

from src.core.file_access import (
    WS_MAX_FILE_SIZE,
    grep_text_files,
    safe_path,
    scan_directory,
)
from src.knowledge.store import KnowledgeStore
from src.memory.database import get_engine, get_session_maker

mcp = FastMCP("zhvusha-knowledge")

# Lazy-initialized store (created on first tool call)
_store: KnowledgeStore | None = None
_store_lock = asyncio.Lock()
_engine_ref: Any = None
_session_maker_ref: async_sessionmaker[AsyncSession] | None = None


def _get_database_url() -> str:
    """Get database URL from environment. Fails if not set."""
    import os

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        msg = "DATABASE_URL environment variable is required for MCP server"
        raise RuntimeError(msg)
    return url


async def _get_store() -> KnowledgeStore:
    """Get or create the KnowledgeStore singleton."""
    global _store, _engine_ref, _session_maker_ref
    if _store is not None:
        return _store

    async with _store_lock:
        if _store is not None:
            return _store

        engine = get_engine(_get_database_url())
        _engine_ref = engine
        session_maker = get_session_maker(engine)
        _session_maker_ref = session_maker
        _store = KnowledgeStore(session_maker)
        return _store


async def _get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get the async session maker, initializing if needed."""
    if _session_maker_ref is not None:
        return _session_maker_ref
    async with _store_lock:
        if _session_maker_ref is not None:
            return _session_maker_ref
        await _get_store()
        assert _session_maker_ref is not None  # set by _get_store
        return _session_maker_ref


def _cleanup_engine() -> None:
    """Dispose SQLAlchemy engine on process exit."""
    if _engine_ref is not None:
        _engine_ref.sync_engine.dispose()


atexit.register(_cleanup_engine)


def _format_json(data: Any) -> str:
    """Format data as readable JSON."""
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
async def search_knowledge(
    query: str,
    category: str | None = None,
    tags: str | None = None,
    limit: int = 10,
) -> str:
    """Search the knowledge base using hybrid semantic + full-text search.

    Returns an index of results (Level 1): id, title, tags, token_count.
    Use get_summaries or get_full_content to read the actual content.

    Args:
        query: Search query in any language.
        category: Optional category path filter (e.g. "tools.libraries").
        tags: Optional comma-separated tags to filter by.
        limit: Max results to return (default 10).
    """
    store = await _get_store()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    results = await store.hybrid_search(
        query, category=category, tags=tag_list, limit=limit
    )

    if not results:
        return "No results found."

    return _format_json([r.model_dump() for r in results])


@mcp.tool()
async def browse_categories(
    parent_path: str | None = None,
) -> str:
    """Browse the knowledge base category tree.

    Without arguments — returns root categories with entry counts.
    With parent_path — returns children of that category.

    Args:
        parent_path: Optional parent category path (e.g. "tools").
    """
    store = await _get_store()
    categories = await store.browse_categories(parent_path)

    if not categories:
        return "No categories found."

    return _format_json([c.model_dump() for c in categories])


@mcp.tool()
async def get_summaries(
    entry_ids: str,
) -> str:
    """Get summaries (Level 2, ~100 tokens each) for selected entries.

    Use after search_knowledge to read entry summaries before
    requesting the full content.

    Args:
        entry_ids: Comma-separated entry IDs (e.g. "42,57,103").
    """
    store = await _get_store()
    ids = [int(x.strip()) for x in entry_ids.split(",")]
    summaries = await store.get_summaries(ids)

    if not summaries:
        return "No entries found for the given IDs."

    return _format_json([s.model_dump() for s in summaries])


@mcp.tool()
async def get_full_content(
    entry_id: int,
) -> str:
    """Get the full text of a knowledge entry (Level 3).

    Use only when the summary is insufficient.

    Args:
        entry_id: The entry ID to retrieve.
    """
    store = await _get_store()
    entry = await store.get_full_content(entry_id)

    if entry is None:
        return f"Entry {entry_id} not found."

    return _format_json(entry.model_dump())


@mcp.tool()
async def add_knowledge(
    title: str,
    content: str,
    category_path: str | None = None,
    tags: str | None = None,
    source: str = "manual",
) -> str:
    """Add a new entry to the knowledge base.

    Writes directly — manual MCP tools don't require staging.
    Automatically generates embedding and estimates token count.

    Args:
        title: Entry title.
        content: Full text content.
        category_path: Optional category (e.g. "tools.libraries").
        tags: Optional comma-separated tags.
        source: Where the knowledge came from (default "manual").
    """
    store = await _get_store()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    entry_id = await store.add_entry(
        title=title,
        content=content,
        category_path=category_path,
        tags=tag_list,
        source=source,
    )

    return f"Entry #{entry_id} added: {title}"


@mcp.tool()
async def list_pending_staging(limit: int = 20) -> str:
    """List pending knowledge staging proposals awaiting review.

    Shows proposals from the sleep agent, MCP, or daemon that need
    approval before being applied to the knowledge base.

    Args:
        limit: Max items to return (default 20).
    """
    store = await _get_store()
    items = await store.get_pending_staged(limit=limit)

    if not items:
        return "No pending proposals."

    result = []
    for item in items:
        result.append(
            {
                "id": item.id,
                "operation": item.operation,
                "target_entry_id": item.target_entry_id,
                "proposed_changes": item.proposed_changes,
                "reason": item.reason,
                "proposed_by": item.proposed_by,
                "created_at": str(item.created_at) if item.created_at else None,
            }
        )

    return _format_json(result)


@mcp.tool()
async def review_staging(staging_id: int, approve: bool) -> str:
    """Approve or reject a pending knowledge staging proposal.

    Approved proposals are applied immediately to the knowledge base.
    Rejected proposals are archived without changes.

    Args:
        staging_id: The staging item ID (from list_pending_staging).
        approve: True to approve and apply, False to reject.
    """
    store = await _get_store()
    ok = await store.review_staged(staging_id, approve=approve)

    if not ok:
        return f"Staging #{staging_id} not found or already reviewed."

    action = "Approved and applied" if approve else "Rejected"
    return f"{action}: staging #{staging_id}."


@mcp.tool()
async def archive_knowledge(entry_id: int) -> str:
    """Archive (soft-delete) a knowledge entry.

    The entry is not physically deleted — its status changes to 'archived'
    and it stops appearing in search results.

    Args:
        entry_id: The entry ID to archive.
    """
    store = await _get_store()
    ok = await store.archive_entry(entry_id)

    if not ok:
        return f"Entry {entry_id} not found or already archived."

    return f"Archived: entry #{entry_id}."


# ------------------------------------------------------------------ #
#  Workspace tools                                                     #
# ------------------------------------------------------------------ #


@functools.cache
def _ws_root() -> Path:
    """Get workspace root path from env."""
    raw = os.environ.get("WORKSPACE_PATH", "~/zhvusha-workspace")
    return Path(raw).expanduser()


def _ws_safe(requested: str) -> Path | None:
    """Resolve workspace path safely (prevents traversal)."""
    return safe_path(_ws_root(), requested)


def _atomic_write_text(target: Path, content: str) -> None:
    """Atomically write content to target via temp file + replace."""
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o644)
        with open(fd, "w", encoding="utf-8", closefd=False) as f:
            f.write(content)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    finally:
        os.close(fd)
    tmp.replace(target)


@mcp.tool()
async def list_workspace(path: str | None = None) -> str:
    """List files and directories in the workspace.

    Without arguments — lists the workspace root.
    With path — lists contents of that subdirectory.

    Args:
        path: Optional relative path within workspace (e.g. "diary" or "inbox").
    """
    ws = _ws_root()
    if not ws.exists():
        return "Workspace directory not found."

    if path:
        target = _ws_safe(path)
        if target is None:
            return "Invalid path."
    else:
        target = ws

    if not target.exists():
        return f"Path not found: {path}"

    if not target.is_dir():
        return f"Not a directory: {path}"

    try:
        items = await asyncio.to_thread(scan_directory, target, ws)
    except PermissionError:
        return f"Permission denied: {path or '/'}"
    except OSError as exc:
        return f"Error reading directory: {exc}"
    if not items:
        return "Directory is empty."
    return _format_json(items)


@mcp.tool()
async def read_workspace_file(path: str) -> str:
    """Read a file from the workspace.

    Returns the text content of a workspace file. Max 2 MB.
    Common paths: diary/, channel/posts/, inbox/, personality/.

    Args:
        path: Relative path to the file (e.g. "diary/2025-04-07.md").
    """
    resolved = _ws_safe(path)
    if resolved is None or not resolved.is_file():
        return f"File not found: {path}"

    try:
        if resolved.stat().st_size > WS_MAX_FILE_SIZE:
            return f"File too large (>{WS_MAX_FILE_SIZE // 1024 // 1024} MB): {path}"
    except OSError:
        return f"File not found: {path}"

    try:
        content = await asyncio.to_thread(
            resolved.read_text, encoding="utf-8", errors="replace"
        )
    except OSError as e:
        return f"Read error: {e}"

    return content


@mcp.tool()
async def search_workspace(
    query: str,
    path: str | None = None,
    max_results: int = 20,
) -> str:
    """Search workspace files by text content (grep).

    Searches through text files in the workspace for lines matching the query.

    Args:
        query: Text to search for (case-insensitive substring match).
        path: Optional subdirectory to search in (e.g. "diary").
        max_results: Maximum number of matching lines to return (default 20).
    """
    ws = _ws_root()
    if path:
        search_root = _ws_safe(path)
        if search_root is None or not search_root.is_dir():
            return f"Invalid search path: {path}"
    else:
        search_root = ws

    if not search_root.exists():
        return "Workspace directory not found."

    limit = max(1, min(max_results, 100))
    matches = await asyncio.to_thread(
        grep_text_files, search_root, ws, query.lower(), limit
    )
    if not matches:
        return f'No matches for "{query}".'
    return _format_json(matches)


@mcp.tool()
async def write_workspace_file(
    path: str,
    content: str,
    mode: str = "create",
) -> str:
    """Write a file to the workspace.

    Modes:
    - "create": Create a new file. Fails if file already exists.
    - "overwrite": Overwrite an existing file or create a new one.

    For appending to existing files, use append_to_workspace_file instead.

    Args:
        path: Relative path for the file (e.g. "diary/2025-04-07.md").
        content: Text content to write.
        mode: Write mode — "create" or "overwrite".
    """
    if mode not in ("create", "overwrite"):
        return f'Invalid mode: "{mode}". Use "create" or "overwrite".'

    if len(content.encode("utf-8")) > WS_MAX_FILE_SIZE:
        return f"Content too large (>{WS_MAX_FILE_SIZE // 1024 // 1024} MB)."

    resolved = _ws_safe(path)
    if resolved is None:
        return "Invalid path (traversal detected)."
    if resolved.is_dir():
        return f"Path is a directory: {path}"

    def _write() -> str:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if mode == "create":
            try:
                with open(resolved, "x", encoding="utf-8") as f:  # noqa: PTH123
                    f.write(content)
            except FileExistsError:
                return f"File already exists: {path}. Use mode='overwrite' to replace."
        else:
            _atomic_write_text(resolved, content)
        return f"Written: {path} ({len(content)} chars, mode={mode})"

    try:
        return await asyncio.to_thread(_write)
    except OSError as e:
        return f"Write error: {e}"


@mcp.tool()
async def append_to_workspace_file(
    path: str,
    content: str,
    separator: str = "\n\n",
) -> str:
    """Append content to a workspace file with a separator.

    If the file doesn't exist, it will be created (without separator prefix).

    Args:
        path: Relative path to the file (e.g. "diary/2025-04-07.md").
        content: Text content to append.
        separator: Separator inserted before the new content (default "\\n\\n").
    """
    if len(content.encode("utf-8")) > WS_MAX_FILE_SIZE:
        return f"Content too large (>{WS_MAX_FILE_SIZE // 1024 // 1024} MB)."

    resolved = _ws_safe(path)
    if resolved is None:
        return "Invalid path (traversal detected)."
    if resolved.is_dir():
        return f"Path is a directory: {path}"

    def _append() -> str:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists():
            existing = resolved.read_text(encoding="utf-8", errors="replace")
            total = (
                len(existing.encode("utf-8"))
                + len(separator.encode("utf-8"))
                + len(content.encode("utf-8"))
            )
            if total > WS_MAX_FILE_SIZE:
                return (
                    f"File would exceed {WS_MAX_FILE_SIZE // 1024 // 1024} MB"
                    " after append."
                )
            new_content = existing + separator + content
        else:
            new_content = content
        _atomic_write_text(resolved, new_content)
        return f"Appended to: {path} ({len(content)} chars)"

    try:
        return await asyncio.to_thread(_append)
    except OSError as e:
        return f"Write error: {e}"


@mcp.tool()
async def delete_workspace_file(
    path: str,
    confirm: bool = False,
) -> str:
    """Delete a file from the workspace.

    Requires confirm=True to actually delete. Without it, returns
    a confirmation prompt instead.

    Args:
        path: Relative path to the file (e.g. "inbox/old_notes.md").
        confirm: Must be True to actually delete. Safety guard.
    """
    resolved = _ws_safe(path)
    if resolved is None:
        return "Invalid path (traversal detected)."

    if not resolved.is_file():
        return f"File not found: {path}"

    if not confirm:
        return (
            f"Are you sure you want to delete '{path}'? "
            "Call again with confirm=True to proceed."
        )

    def _delete() -> str:
        try:
            resolved.unlink()
        except FileNotFoundError:
            return f"File not found: {path}"
        return f"Deleted: {path}"

    try:
        return await asyncio.to_thread(_delete)
    except OSError as e:
        return f"Delete error: {e}"


# ------------------------------------------------------------------ #
#  Dashboard routes (registered via custom_route)                      #
# ------------------------------------------------------------------ #


def register_dashboard_routes() -> None:
    """Register all dashboard HTTP routes on the MCP server."""
    from src.mcp_server.dashboard_api import (
        api_daemon_status,
        api_episodes,
        api_knowledge_entry,
        api_knowledge_graph,
        api_knowledge_tree,
        api_staging,
        api_workspace_file,
        api_workspace_tree,
        dashboard_page,
    )

    mcp.custom_route("/dashboard", methods=["GET"])(dashboard_page)
    mcp.custom_route("/api/knowledge/graph", methods=["GET"])(api_knowledge_graph)
    mcp.custom_route("/api/knowledge/tree", methods=["GET"])(api_knowledge_tree)
    mcp.custom_route("/api/knowledge/entry/{entry_id:int}", methods=["GET"])(
        api_knowledge_entry
    )
    mcp.custom_route("/api/workspace/tree", methods=["GET"])(api_workspace_tree)
    mcp.custom_route("/api/workspace/file", methods=["GET"])(api_workspace_file)
    mcp.custom_route("/api/episodes", methods=["GET"])(api_episodes)
    mcp.custom_route("/api/staging", methods=["GET"])(api_staging)
    mcp.custom_route("/api/daemon/status", methods=["GET"])(api_daemon_status)
