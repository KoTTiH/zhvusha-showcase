"""Generate adversarial test drafts anchored to real archive nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.archive.models import ArchiveNode, ArchiveStatus

if TYPE_CHECKING:
    from src.archive.store import ArchiveStore

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AdversarialTestDraft:
    """A proposed hard test tied back to a concrete archive failure."""

    archive_slug: str
    test_file: str
    test_name: str
    body: str
    rationale: str
    tags: list[str] = field(default_factory=list)


def generate_adversarial_tests(
    nodes: list[ArchiveNode], *, limit: int = 5
) -> list[AdversarialTestDraft]:
    """Create test drafts from failed archive nodes.

    The anchor is non-negotiable: every generated draft cites the source
    archive node in the body so the test cannot drift into synthetic puzzles
    unrelated to Жвуша's real failures.
    """
    drafts: list[AdversarialTestDraft] = []
    for node in nodes:
        if node.status != ArchiveStatus.FAILED:
            continue
        drafts.append(_draft_from_node(node))
        if len(drafts) >= limit:
            break
    return drafts


class ArchiveAdversarialTestProvider:
    """Generate adversarial drafts directly from archive lookup."""

    def __init__(self, store: ArchiveStore | None) -> None:
        self._store = store

    async def generate(
        self, query: str, *, limit: int = 5
    ) -> list[AdversarialTestDraft]:
        if self._store is None:
            return []
        nodes = await self._store.lookup(query, top_k=max(limit * 2, 10))
        return generate_adversarial_tests(nodes, limit=limit)


def _draft_from_node(node: ArchiveNode) -> AdversarialTestDraft:
    base = _slugify(node.spec_slug or node.slug)
    test_name = f"test_adversarial_{base.replace('-', '_')}"
    test_file = f"tests/adversarial/test_{base}.py"
    body = (
        f"def {test_name}():\n"
        f'    """Regression anchored to archive node {node.slug}."""\n'
        f"    archive_node = {node.slug!r}\n"
        f"    failure_pattern = {node.insight[:500]!r}\n"
        f"    assert archive_node\n"
        f"    assert failure_pattern\n"
    )
    return AdversarialTestDraft(
        archive_slug=node.slug,
        test_file=test_file,
        test_name=test_name,
        body=body,
        rationale=(
            "Generated from a real failed self-coding cycle; keep the "
            "archive node reference when converting this draft into a spec."
        ),
        tags=list(node.tags),
    )


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug[:80] or "archive-failure"
