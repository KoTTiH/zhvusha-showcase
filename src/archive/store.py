"""Database store and deterministic lookup for self-coding archive nodes."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import text

from src.archive.models import ArchiveNode, ArchiveStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_TOKEN_RE = re.compile(r"[a-zа-я0-9_]+", re.IGNORECASE)


class ArchiveStore:
    """Async SQL store for ``archive_nodes``."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def upsert(self, node: ArchiveNode) -> None:
        async with self._session_maker() as session:
            await session.execute(_UPSERT, _params(node))
            await session.commit()

    async def get(self, slug: str) -> ArchiveNode | None:
        async with self._session_maker() as session:
            row = (
                (await session.execute(_GET_BY_SLUG, {"slug": slug})).mappings().first()
            )
        return _row_to_node(dict(row)) if row is not None else None

    async def lookup(self, query: str, *, top_k: int = 5) -> list[ArchiveNode]:
        async with self._session_maker() as session:
            rows = [
                dict(row)
                for row in (
                    await session.execute(_GET_RECENT, {"limit": max(top_k * 20, 50)})
                ).mappings()
            ]
        return archive_lookup(
            query,
            [_row_to_node(row) for row in rows],
            top_k=top_k,
        )


def archive_lookup(
    query: str, nodes: list[ArchiveNode], *, top_k: int = 5
) -> list[ArchiveNode]:
    """Rank archive nodes by deterministic token overlap.

    This is intentionally simple until pgvector embeddings are wired. It keeps
    the contract useful in tests and offline local runs without adding another
    runtime dependency path.
    """

    wanted = _tokens(query)
    if not wanted:
        return sorted(nodes, key=lambda node: node.created_at, reverse=True)[:top_k]
    scored = [
        (_score(wanted, node), node.created_at, node)
        for node in nodes
        if _score(wanted, node) > 0
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [node for _score_value, _created_at, node in scored[:top_k]]


def _params(node: ArchiveNode) -> dict[str, Any]:
    return {
        "slug": node.slug,
        "spec_slug": node.spec_slug,
        "proposal_slug": node.proposal_slug,
        "tier": node.tier,
        "status": node.status.value,
        "created_at": node.created_at,
        "branch": node.branch,
        "commit_sha": node.commit_sha,
        "parent_slug": node.parent_slug,
        "diff_summary": node.diff_summary,
        "tests_summary": node.tests_summary,
        "rationale": node.rationale,
        "insight": node.insight,
        "source_evidence": _json_param(node.source_evidence),
        "model_config": _json_param(node.runtime_config),
        "tags": _json_param(node.tags),
        "metadata": _json_param(node.metadata),
    }


def _row_to_node(row: dict[str, Any]) -> ArchiveNode:
    created_at = row.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    evidence = row.get("source_evidence")
    model_config = row.get("model_config")
    tags = row.get("tags")
    metadata = row.get("metadata")
    decoded_evidence = _json_value(evidence)
    decoded_model_config = _json_value(model_config)
    decoded_tags = _json_value(tags)
    decoded_metadata = _json_value(metadata)
    return ArchiveNode(
        slug=str(row["slug"]),
        spec_slug=cast("str | None", row.get("spec_slug")),
        proposal_slug=cast("str | None", row.get("proposal_slug")),
        tier=cast("Any", int(row["tier"])),
        status=ArchiveStatus(str(row["status"])),
        created_at=cast("datetime", created_at),
        branch=cast("str | None", row.get("branch")),
        commit_sha=cast("str | None", row.get("commit_sha")),
        parent_slug=cast("str | None", row.get("parent_slug")),
        diff_summary=str(row["diff_summary"]),
        tests_summary=str(row["tests_summary"]),
        rationale=str(row.get("rationale") or ""),
        insight=str(row["insight"]),
        source_evidence=cast(
            "list[dict[str, str]]",
            decoded_evidence if isinstance(decoded_evidence, list) else [],
        ),
        model_config=cast(
            "dict[str, str]",
            decoded_model_config if isinstance(decoded_model_config, dict) else {},
        ),
        tags=cast("list[str]", decoded_tags if isinstance(decoded_tags, list) else []),
        metadata=cast(
            "dict[str, Any]",
            decoded_metadata if isinstance(decoded_metadata, dict) else {},
        ),
    )


def _json_param(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _score(wanted: set[str], node: ArchiveNode) -> int:
    haystack = " ".join(
        [
            node.slug,
            node.spec_slug or "",
            node.proposal_slug or "",
            node.diff_summary,
            node.tests_summary,
            node.rationale,
            node.insight,
            " ".join(node.tags),
            json.dumps(node.metadata, ensure_ascii=False, sort_keys=True),
        ]
    )
    return len(wanted & _tokens(haystack))


def _tokens(text_value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text_value)}


_UPSERT = text(
    """
    INSERT INTO archive_nodes (
        slug, spec_slug, proposal_slug, tier, status, created_at, branch,
        commit_sha, parent_slug, diff_summary, tests_summary, rationale,
        insight, source_evidence, model_config, tags, metadata
    )
    VALUES (
        :slug, :spec_slug, :proposal_slug, :tier, :status, :created_at, :branch,
        :commit_sha, :parent_slug, :diff_summary, :tests_summary, :rationale,
        :insight,
        CAST(:source_evidence AS jsonb),
        CAST(:model_config AS jsonb),
        CAST(:tags AS jsonb),
        CAST(:metadata AS jsonb)
    )
    ON CONFLICT (slug) DO UPDATE SET
        spec_slug = EXCLUDED.spec_slug,
        proposal_slug = EXCLUDED.proposal_slug,
        tier = EXCLUDED.tier,
        status = EXCLUDED.status,
        branch = EXCLUDED.branch,
        commit_sha = EXCLUDED.commit_sha,
        parent_slug = EXCLUDED.parent_slug,
        diff_summary = EXCLUDED.diff_summary,
        tests_summary = EXCLUDED.tests_summary,
        rationale = EXCLUDED.rationale,
        insight = EXCLUDED.insight,
        source_evidence = EXCLUDED.source_evidence,
        model_config = EXCLUDED.model_config,
        tags = EXCLUDED.tags,
        metadata = EXCLUDED.metadata
    """
)

_GET_BY_SLUG = text(
    """
    SELECT slug, spec_slug, proposal_slug, tier, status, created_at, branch,
           commit_sha, parent_slug, diff_summary, tests_summary, rationale,
           insight, source_evidence, model_config, tags, metadata
    FROM archive_nodes
    WHERE slug = :slug
    LIMIT 1
    """
)

_GET_RECENT = text(
    """
    SELECT slug, spec_slug, proposal_slug, tier, status, created_at, branch,
           commit_sha, parent_slug, diff_summary, tests_summary, rationale,
           insight, source_evidence, model_config, tags, metadata
    FROM archive_nodes
    ORDER BY created_at DESC
    LIMIT :limit
    """
)
