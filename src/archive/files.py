"""Filesystem artifacts for archived self-coding cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from src.archive.models import ArchiveNode


class ArchiveFileWriter:
    """Write human-auditable files for one archive node."""

    def __init__(self, archive_root: Path) -> None:
        self._archive_root = archive_root

    def write_node(self, node: ArchiveNode) -> Path:
        node_dir = self._archive_root / node.slug
        node_dir.mkdir(parents=True, exist_ok=True)
        _write_text(node_dir / "insight.md", _insight_markdown(node))
        _write_text(node_dir / "rationale.md", node.rationale or "Нет rationale.\n")
        spec_snapshot = node.metadata.get("spec_snapshot")
        if isinstance(spec_snapshot, dict):
            _write_yaml(node_dir / "spec_snapshot.yaml", spec_snapshot)
            chat_context = spec_snapshot.get("chat_context")
            if isinstance(chat_context, list) and chat_context:
                _write_text(
                    node_dir / "chat_context.md",
                    _chat_context_markdown(chat_context),
                )
        _write_yaml(node_dir / "source_evidence.yaml", node.source_evidence)
        _write_yaml(node_dir / "model_config.yaml", node.runtime_config)
        parent_slugs = node.metadata.get("parent_node_slugs")
        if isinstance(parent_slugs, list) and parent_slugs:
            _write_yaml(
                node_dir / "parent_links.yaml",
                {
                    "parent_node_slugs": [
                        slug for slug in parent_slugs if isinstance(slug, str)
                    ]
                },
            )
        _write_yaml(
            node_dir / "metadata.yaml",
            node.model_dump(mode="json", by_alias=True),
        )
        return node_dir


def _insight_markdown(node: ArchiveNode) -> str:
    return (
        f"# {node.slug}\n\n"
        f"status: `{node.status.value}`  \n"
        f"tier: `{node.tier}`  \n"
        f"spec: `{node.spec_slug or '—'}`  \n"
        f"commit: `{node.commit_sha or '—'}`\n\n"
        f"## Что изменилось\n{node.diff_summary}\n\n"
        f"## Проверки\n{node.tests_summary}\n\n"
        f"## Вывод\n{node.insight}\n"
    )


def _chat_context_markdown(lines: list[Any]) -> str:
    body = "\n".join(f"- {line}" for line in lines if isinstance(line, str))
    return "# Контекст /самокодинг\n\n" + (body or "Нет контекста.")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
