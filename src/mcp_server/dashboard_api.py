"""REST API endpoints for the Obsidian-like knowledge dashboard.

All endpoints are registered via FastMCP.custom_route and served
on the same port as the MCP SSE transport.
"""

from __future__ import annotations

import asyncio
import atexit
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from starlette.responses import HTMLResponse, JSONResponse

from src.knowledge.models import (
    Category,
    EntryRelation,
    KnowledgeEntry,
    KnowledgeStagingItem,
)
from src.memory.database import Episode

if TYPE_CHECKING:
    from starlette.requests import Request

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_DASHBOARD_HTML: str | None = None
_redis_client: Any = None
_redis_lock = asyncio.Lock()


def _cleanup_redis() -> None:
    """Best-effort Redis cleanup for process exit."""
    global _redis_client
    if _redis_client is not None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_redis_client.aclose())
            loop.close()
        except Exception:  # noqa: S110 — best-effort cleanup at exit
            pass
        _redis_client = None


atexit.register(_cleanup_redis)


def _get_dashboard_html() -> str:
    """Load and cache the dashboard HTML file."""
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).parent / "static" / "dashboard.html"
        _DASHBOARD_HTML = html_path.read_text(encoding="utf-8")
    return _DASHBOARD_HTML


def _workspace_path() -> Path:
    """Resolve workspace path from env or default."""
    raw = os.environ.get("WORKSPACE_PATH", "~/zhvusha-workspace")
    return Path(raw).expanduser()


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(
        data,
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


def _safe_path(base: Path, requested: str) -> Path | None:
    """Resolve requested path under base, prevent traversal."""
    from src.core.file_access import safe_path

    return safe_path(base, requested)


# ---------------------------------------------------------------------------
#  Route handlers
# ---------------------------------------------------------------------------


async def dashboard_page(request: Request) -> HTMLResponse:  # noqa: ARG001
    """Serve the single-page dashboard HTML."""
    return HTMLResponse(_get_dashboard_html())


async def api_knowledge_graph(request: Request) -> JSONResponse:  # noqa: ARG001, C901
    """Return all knowledge nodes + edges for D3.js graph.

    Edges include explicit entry_relations AND implicit tag-overlap edges.
    """
    from src.mcp_server.server import _get_store

    store = await _get_store()

    async with store.session() as session:
        # Fetch all non-archived entries
        entries_result = await session.execute(
            select(
                KnowledgeEntry.id,
                KnowledgeEntry.title,
                KnowledgeEntry.tags,
                KnowledgeEntry.content_type,
                KnowledgeEntry.status,
                KnowledgeEntry.token_count,
                KnowledgeEntry.category_id,
                KnowledgeEntry.created_at,
            ).where(KnowledgeEntry.status != "archived")
        )
        entries = entries_result.fetchall()

        # Fetch category info for each entry
        cat_result = await session.execute(
            select(Category.id, Category.path, Category.name_ru)
        )
        categories = {r[0]: (r[1], r[2]) for r in cat_result.fetchall()}

        # Fetch explicit relations
        rels_result = await session.execute(
            select(
                EntryRelation.source_id,
                EntryRelation.target_id,
                EntryRelation.relation_type,
            )
        )
        explicit_edges = rels_result.fetchall()

    # Build nodes
    nodes: list[dict[str, Any]] = []
    # Map entry_id -> set of tags for tag-overlap computation
    tag_map: dict[int, set[str]] = {}
    entry_ids: set[int] = set()

    for e in entries:
        eid, title, tags, content_type, _status, token_count, cat_id, created_at = e
        cat_path, cat_name_ru = (
            categories.get(cat_id, (None, None)) if cat_id else (None, None)
        )

        # Determine node type from category path
        node_type = "knowledge"
        if cat_path:
            root = cat_path.split(".")[0]
            type_map = {
                "personality": "personality",
                "diary": "diary",
                "episodes": "episodes",
                "journal": "diary",
            }
            node_type = type_map.get(root, "knowledge")

        tag_set = set(tags) if tags else set()
        tag_map[eid] = tag_set
        entry_ids.add(eid)

        nodes.append(
            {
                "id": eid,
                "title": title,
                "tags": tags or [],
                "type": node_type,
                "content_type": content_type,
                "token_count": token_count or 0,
                "category_path": cat_path,
                "category_name_ru": cat_name_ru,
                "created_at": str(created_at) if created_at else None,
                "link_count": 0,
            }
        )

    # Build edges: explicit
    edges: list[dict[str, Any]] = []
    link_counts: dict[int, int] = {}

    for src_id, tgt_id, rel_type in explicit_edges:
        if src_id in entry_ids and tgt_id in entry_ids:
            edges.append(
                {"source": src_id, "target": tgt_id, "type": rel_type or "related"}
            )
            link_counts[src_id] = link_counts.get(src_id, 0) + 1
            link_counts[tgt_id] = link_counts.get(tgt_id, 0) + 1

    # Build edges: tag overlap (shared tags = edge) — O(n²), offloaded to thread
    def _compute_tag_edges() -> None:
        id_list = list(tag_map.keys())
        seen_pairs: set[tuple[int, int]] = {(e["source"], e["target"]) for e in edges}
        for i, a_id in enumerate(id_list):
            a_tags = tag_map[a_id]
            if not a_tags:
                continue
            for b_id in id_list[i + 1 :]:
                b_tags = tag_map[b_id]
                if not b_tags:
                    continue
                common = a_tags & b_tags
                if (
                    common
                    and (a_id, b_id) not in seen_pairs
                    and (b_id, a_id) not in seen_pairs
                ):
                    edges.append(
                        {"source": a_id, "target": b_id, "type": "tag_overlap"}
                    )
                    link_counts[a_id] = link_counts.get(a_id, 0) + 1
                    link_counts[b_id] = link_counts.get(b_id, 0) + 1

    await asyncio.to_thread(_compute_tag_edges)

    # Update link_count in nodes
    for node in nodes:
        node["link_count"] = link_counts.get(node["id"], 0)

    return _json({"nodes": nodes, "edges": edges})


async def api_knowledge_tree(request: Request) -> JSONResponse:  # noqa: ARG001
    """Return category tree with entry counts and entries as leaves."""
    from src.mcp_server.server import _get_store

    store = await _get_store()

    async with store.session() as session:
        result = await session.execute(
            select(
                Category.id,
                Category.name,
                Category.name_ru,
                Category.path,
                Category.entry_count,
                Category.parent_id,
            ).order_by(Category.path)
        )
        rows = result.fetchall()

        # Fetch entries per category
        entries_result = await session.execute(
            select(
                KnowledgeEntry.id,
                KnowledgeEntry.title,
                KnowledgeEntry.category_id,
                KnowledgeEntry.token_count,
                KnowledgeEntry.content_type,
            )
            .where(KnowledgeEntry.status != "archived")
            .where(KnowledgeEntry.category_id.isnot(None))
            .order_by(KnowledgeEntry.title)
        )
        all_entries = entries_result.fetchall()

    # Group entries by category_id
    entries_by_cat: dict[int, list[dict[str, Any]]] = {}
    for e in all_entries:
        eid, title, cat_id, token_count, content_type = e
        entries_by_cat.setdefault(cat_id, []).append(
            {
                "id": eid,
                "title": title,
                "token_count": token_count or 0,
                "content_type": content_type,
                "type": "entry",
            }
        )

    # Build tree structure
    all_cats: list[dict[str, Any]] = []
    for r in rows:
        all_cats.append(
            {
                "id": r[0],
                "name": r[1],
                "name_ru": r[2],
                "path": r[3],
                "entry_count": r[4],
                "parent_id": r[5],
                "children": [],
                "entries": entries_by_cat.get(r[0], []),
            }
        )

    # Index by id
    by_id = {c["id"]: c for c in all_cats}
    roots: list[dict[str, Any]] = []
    for c in all_cats:
        pid = c["parent_id"]
        if pid and pid in by_id:
            by_id[pid]["children"].append(c)
        else:
            roots.append(c)

    return _json(roots)


async def api_knowledge_entry(request: Request) -> JSONResponse:
    """Return full content of a knowledge entry + backlinks."""
    entry_id = int(request.path_params["entry_id"])

    from src.mcp_server.server import _get_store

    store = await _get_store()

    async with store.session() as session:
        result = await session.execute(
            select(KnowledgeEntry)
            .where(KnowledgeEntry.id == entry_id)
            .options(selectinload(KnowledgeEntry.category))
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            return _json({"error": "not found"}, 404)

        cat_path = entry.category.path if entry.category else None
        cat_name = entry.category.name_ru if entry.category else None

        # Forward links
        fwd = await session.execute(
            select(EntryRelation.target_id, EntryRelation.relation_type).where(
                EntryRelation.source_id == entry_id
            )
        )
        forward_links = [
            {"id": r[0], "type": r[1] or "related"} for r in fwd.fetchall()
        ]

        # Backlinks
        back = await session.execute(
            select(EntryRelation.source_id, EntryRelation.relation_type).where(
                EntryRelation.target_id == entry_id
            )
        )
        backlinks = [{"id": r[0], "type": r[1] or "related"} for r in back.fetchall()]

        # Get titles for linked entries
        linked_ids = [lnk["id"] for lnk in forward_links + backlinks]
        titles_map: dict[int, str] = {}
        if linked_ids:
            titles_result = await session.execute(
                select(KnowledgeEntry.id, KnowledgeEntry.title).where(
                    KnowledgeEntry.id.in_(linked_ids)
                )
            )
            titles_map = {r[0]: r[1] for r in titles_result.fetchall()}

        for link in forward_links + backlinks:
            link["title"] = titles_map.get(link["id"], f"#{link['id']}")

        data = {
            "id": entry.id,
            "title": entry.title,
            "content": entry.content,
            "summary": entry.summary,
            "tags": entry.tags,
            "source": entry.source,
            "source_url": entry.source_url,
            "content_type": entry.content_type,
            "status": entry.status,
            "token_count": entry.token_count,
            "category_path": cat_path,
            "category_name_ru": cat_name,
            "created_at": str(entry.created_at) if entry.created_at else None,
            "updated_at": str(entry.updated_at) if entry.updated_at else None,
            "forward_links": forward_links,
            "backlinks": backlinks,
        }

    return _json(data)


async def api_workspace_tree(request: Request) -> JSONResponse:  # noqa: ARG001
    """Return file tree from workspace directory."""
    ws = _workspace_path()
    if not ws.exists():
        return _json([])

    def scan_dir(path: Path, depth: int = 0) -> list[dict[str, Any]]:
        if depth > 5:
            return []
        items: list[dict[str, Any]] = []
        try:
            for child in sorted(path.iterdir()):
                if child.name.startswith(".") or child.is_symlink():
                    continue
                if child.is_dir():
                    items.append(
                        {
                            "name": child.name,
                            "type": "dir",
                            "path": str(child.relative_to(ws)),
                            "children": scan_dir(child, depth + 1),
                        }
                    )
                else:
                    items.append(
                        {
                            "name": child.name,
                            "type": "file",
                            "path": str(child.relative_to(ws)),
                            "size": child.stat().st_size,
                        }
                    )
        except PermissionError:
            pass
        return items

    return _json(await asyncio.to_thread(scan_dir, ws))


async def api_workspace_file(request: Request) -> JSONResponse:
    """Return content of a workspace file."""
    file_path = request.query_params.get("path", "")
    if not file_path:
        return _json({"error": "path required"}, 400)

    ws = _workspace_path()
    resolved = _safe_path(ws, file_path)
    if resolved is None or not resolved.is_file():
        return _json({"error": "not found"}, 404)

    from src.core.file_access import WS_MAX_FILE_SIZE

    if resolved.stat().st_size > WS_MAX_FILE_SIZE:
        return _json({"error": "file too large"}, 413)

    try:
        content = await asyncio.to_thread(
            resolved.read_text, encoding="utf-8", errors="replace"
        )
    except OSError:
        return _json({"error": "read error"}, 500)

    return _json({"path": file_path, "content": content, "size": len(content)})


async def api_episodes(request: Request) -> JSONResponse:
    """Return recent episodes."""
    try:
        limit = int(request.query_params.get("limit", "50"))
    except (ValueError, TypeError):
        return _json({"error": "invalid limit parameter"}, 400)
    limit = max(1, min(limit, 200))

    from src.mcp_server.server import _get_session_maker

    session_maker = await _get_session_maker()
    async with session_maker() as session:
        result = await session.execute(
            select(
                Episode.id,
                Episode.timestamp,
                Episode.role,
                Episode.content,
                Episode.summary,
                Episode.importance,
                Episode.valence,
                Episode.source,
                Episode.intent,
                Episode.emotion,
            )
            .order_by(Episode.timestamp.desc())
            .limit(limit)
        )
        rows = result.fetchall()

    episodes = [
        {
            "id": r[0],
            "timestamp": str(r[1]) if r[1] else None,
            "role": r[2],
            "content": r[3][:300] if r[3] else None,
            "summary": r[4],
            "importance": r[5],
            "valence": r[6],
            "source": r[7],
            "intent": r[8],
            "emotion": r[9],
        }
        for r in rows
    ]

    return _json(episodes)


async def api_staging(request: Request) -> JSONResponse:  # noqa: ARG001
    """Return pending staging proposals."""
    from src.mcp_server.server import _get_store

    store = await _get_store()

    items = await store.get_pending_staged(limit=50)
    result = [
        {
            "id": item.id,
            "operation": item.operation,
            "target_entry_id": item.target_entry_id,
            "proposed_changes": item.proposed_changes,
            "reason": item.reason,
            "proposed_by": item.proposed_by,
            "created_at": str(item.created_at) if item.created_at else None,
        }
        for item in items
    ]

    return _json(result)


async def api_daemon_status(request: Request) -> JSONResponse:  # noqa: ARG001
    """Return daemon status, Redis queue sizes, LLM budget."""
    import redis.asyncio as aioredis

    from src.core.config import get_settings

    settings = get_settings()
    status: dict[str, Any] = {
        "daemon_enabled": settings.daemon_enabled,
        "budget_limit_usd": settings.daemon_max_llm_cost_per_day_usd,
    }

    # Redis queue sizes (reuse connection)
    global _redis_client
    try:
        if _redis_client is None:
            async with _redis_lock:
                if _redis_client is None:
                    _redis_client = aioredis.from_url(
                        settings.redis_url, decode_responses=True
                    )
        status["queues"] = {
            "critical": await _redis_client.xlen("signals:critical"),
            "normal": await _redis_client.xlen("signals:normal"),
            "background": await _redis_client.xlen("signals:background"),
        }
    except Exception:
        status["queues"] = {"critical": 0, "normal": 0, "background": 0}

    # LLM budget from usage tracker
    try:
        from src.monitoring.usage_tracker import UsageTracker

        data_dir = Path(settings.workspace_path).expanduser() / "monitoring"
        if data_dir.exists():
            tracker = UsageTracker(data_dir)
            today = tracker.get_today()
            status["budget_spent_usd"] = round(today.cost_usd, 4)
            status["budget_month_usd"] = round(tracker.get_month_total(), 4)
            status["api_calls_today"] = today.total_api
        else:
            status["budget_spent_usd"] = 0
            status["budget_month_usd"] = 0
            status["api_calls_today"] = 0
    except Exception:
        status["budget_spent_usd"] = 0
        status["budget_month_usd"] = 0
        status["api_calls_today"] = 0

    # DB stats
    try:
        from src.mcp_server.server import _get_store

        store = await _get_store()
        async with store.session() as session:
            total = await session.execute(
                select(func.count(KnowledgeEntry.id)).where(
                    KnowledgeEntry.status != "archived"
                )
            )
            status["total_entries"] = total.scalar() or 0

            staging_count = await session.execute(
                select(func.count(KnowledgeStagingItem.id)).where(
                    KnowledgeStagingItem.status == "pending"
                )
            )
            status["staging_pending"] = staging_count.scalar() or 0

            episode_count = await session.execute(select(func.count(Episode.id)))
            status["total_episodes"] = episode_count.scalar() or 0

            # Per-category counts
            cat_counts = await session.execute(
                select(Category.name_ru, Category.entry_count)
                .where(Category.entry_count > 0)
                .order_by(Category.entry_count.desc())
            )
            status["categories"] = [
                {"name": r[0], "count": r[1]} for r in cat_counts.fetchall()
            ]
    except Exception:
        logger.warning("dashboard_db_stats_failed", exc_info=True)

    return _json(status)
