"""Create archive nodes after ImplementSpec cycles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.archive.files import ArchiveFileWriter
from src.archive.models import ArchiveNode, ArchiveStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.archive.store import ArchiveStore
    from src.knowledge import KnowledgeStoreProtocol
    from src.skills.spec_command.parser import SpecModel

logger = structlog.get_logger()

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SkillDraftOpportunity:
    """Repeated archive pattern that should become a staged skill spec."""

    pattern: str
    source_archive_slugs: list[str]
    rationale: str


class CycleAnalyzer:
    """Record success/failure evidence for later self-improvement lookup."""

    def __init__(
        self,
        *,
        archive_root: Path,
        store: ArchiveStore | None = None,
        knowledge_store: KnowledgeStoreProtocol | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._files = ArchiveFileWriter(archive_root)
        self._store = store
        self._knowledge_store = knowledge_store
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    async def record_success(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        branch_name: str,
        commit_sha: str,
        sdk_summary: str,
        backend: str,
    ) -> ArchiveNode:
        node = ArchiveNode(
            slug=_node_slug(spec.slug, commit_sha),
            spec_slug=spec.slug,
            proposal_slug=None,
            tier=spec.tier,
            status=ArchiveStatus.COMMITTED,
            created_at=self._clock(),
            branch=branch_name,
            commit_sha=commit_sha,
            parent_slug=_parent_slug(spec),
            diff_summary=sdk_summary or f"Spec {spec.slug} implemented.",
            tests_summary="Commit gate passed: tests, style and types.",
            rationale=spec.rationale,
            insight=_success_insight(spec.slug, sdk_summary),
            source_evidence=_source_evidence(spec),
            model_config={"backend": backend, "executor": "codex_cli"},
            tags=[spec.slug, f"tier-{spec.tier}", "self-coding"],
            metadata=_archive_metadata(
                spec=spec,
                spec_path=spec_path,
                status_source="implement_spec_success",
                backend=backend,
            ),
        )
        await self._persist(node)
        return node

    async def record_failure(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        branch_name: str | None,
        reason: str,
        backend: str = "codex_cli",
    ) -> ArchiveNode:
        node = ArchiveNode(
            slug=_node_slug(spec.slug, f"failed-{self._clock().timestamp():.0f}"),
            spec_slug=spec.slug,
            proposal_slug=None,
            tier=spec.tier,
            status=ArchiveStatus.FAILED,
            created_at=self._clock(),
            branch=branch_name,
            commit_sha=None,
            parent_slug=_parent_slug(spec),
            diff_summary="Cycle stopped before a valid implementation commit.",
            tests_summary=reason[:500] or "Unknown failure.",
            rationale=spec.rationale,
            insight=_failure_insight(spec.slug, reason),
            source_evidence=_source_evidence(spec),
            model_config={"backend": backend, "executor": "codex_cli"},
            tags=[spec.slug, f"tier-{spec.tier}", "self-coding", "failed"],
            metadata=_archive_metadata(
                spec=spec,
                spec_path=spec_path,
                status_source="implement_spec_failure",
                backend=backend,
            ),
        )
        await self._persist(node)
        return node

    async def _persist(self, node: ArchiveNode) -> None:
        node_dir = self._files.write_node(node)
        if self._store is not None:
            await self._store.upsert(node)
        await self._persist_cycle_insight_to_kb(node)
        logger.info("cycle_archived", slug=node.slug, node_dir=str(node_dir))

    async def _persist_cycle_insight_to_kb(self, node: ArchiveNode) -> None:
        if self._knowledge_store is None:
            return
        try:
            await self._knowledge_store.add_entry(
                title=f"Self-coding cycle insight: {node.slug}",
                content=_knowledge_content(node),
                category_path="dev.cycle_insights",
                tags=[*node.tags, node.status.value, "cycle-insight"],
                source="cycle_analyzer",
                source_url=f"archive:{node.slug}",
                content_type="cycle_insight",
                metadata={
                    "archive_slug": node.slug,
                    "spec_slug": node.spec_slug,
                    "commit_sha": node.commit_sha,
                    "parent_node_slugs": list(
                        node.metadata.get("parent_node_slugs", [])
                    ),
                    "backend": node.runtime_config.get("backend", "codex_cli"),
                },
            )
        except Exception:
            logger.warning(
                "cycle_insight_kb_write_failed", slug=node.slug, exc_info=True
            )


def _source_evidence(spec: SpecModel) -> list[dict[str, str]]:
    return [
        {
            "url": source.url,
            "source_type": source.source_type,
            "trust_tier": source.trust_tier,
            "claim": source.claim,
        }
        for source in spec.source_provenance
    ]


def _archive_metadata(
    *,
    spec: SpecModel,
    spec_path: Path,
    status_source: str,
    backend: str,
) -> dict[str, Any]:
    return {
        "status_source": status_source,
        "spec_path": spec_path.as_posix(),
        "spec_snapshot": spec.model_dump(mode="json"),
        "chat_context": list(spec.chat_context),
        "self_coding_actor": "zhvusha",
        "commit_author_name": "zhvusha-coder",
        "commit_author_email": "zhvusha@local",
        "agent_backend": backend,
        "parent_node_slugs": _parent_slugs(spec),
    }


def _parent_slugs(spec: SpecModel) -> list[str]:
    return [attempt.archive_slug for attempt in spec.previous_attempts]


def _parent_slug(spec: SpecModel) -> str | None:
    slugs = _parent_slugs(spec)
    return slugs[0] if slugs else None


def _knowledge_content(node: ArchiveNode) -> str:
    parent_slugs = node.metadata.get("parent_node_slugs")
    parent_line = ", ".join(parent_slugs) if isinstance(parent_slugs, list) else ""
    return (
        f"Archive node: {node.slug}\n"
        f"Status: {node.status.value}\n"
        f"Tier: {node.tier}\n"
        f"Spec: {node.spec_slug or 'none'}\n"
        f"Commit: {node.commit_sha or 'none'}\n"
        f"Parents: {parent_line or 'none'}\n\n"
        f"Diff summary:\n{node.diff_summary}\n\n"
        f"Tests summary:\n{node.tests_summary}\n\n"
        f"Insight:\n{node.insight}\n"
    )


def _success_insight(slug: str, sdk_summary: str) -> str:
    summary = sdk_summary.strip()
    if summary:
        return f"Cycle `{slug}` завершился успешно; следующий похожий spec может переиспользовать этот путь: {summary[:500]}"
    return f"Cycle `{slug}` завершился успешно; commit gate подтвердил результат."


def _failure_insight(slug: str, reason: str) -> str:
    clean = reason.strip() or "неизвестная причина"
    return (
        f"Cycle `{slug}` остановился до commit. В будущих попытках сначала "
        f"проверить этот блокер: {clean[:500]}"
    )


def _node_slug(spec_slug: str, suffix: str) -> str:
    raw = f"{spec_slug}-{suffix[:16]}".lower()
    return _SLUG_RE.sub("-", raw).strip("-")[:120] or "self-coding-cycle"


def detect_skill_draft_opportunity(
    nodes: list[ArchiveNode], *, min_count: int = 3
) -> SkillDraftOpportunity | None:
    """Detect repeated failure/success patterns that deserve a skill draft."""
    buckets: dict[str, list[ArchiveNode]] = {}
    for node in nodes:
        pattern = _pattern_key(node)
        if not pattern:
            continue
        buckets.setdefault(pattern, []).append(node)
    for pattern, grouped in sorted(
        buckets.items(), key=lambda item: len(item[1]), reverse=True
    ):
        if len(grouped) >= min_count:
            return SkillDraftOpportunity(
                pattern=pattern,
                source_archive_slugs=[node.slug for node in grouped[:min_count]],
                rationale=(
                    f"Pattern `{pattern}` appeared in {len(grouped)} archive nodes; "
                    "draft a staged spec instead of repeating manual fixes."
                ),
            )
    return None


def _pattern_key(node: ArchiveNode) -> str:
    for tag in node.tags:
        if tag not in {"self-coding", "failed", f"tier-{node.tier}"}:
            return tag
    tokens = re.findall(r"[a-zа-я0-9]+", node.insight.lower())
    return "-".join(tokens[:2])
