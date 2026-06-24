"""Filesystem store for Tier 3 proposal markdown files."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any

import yaml

from src.skills.proposal_command.models import ProposalModel

if TYPE_CHECKING:
    from pathlib import Path

_FRONTMATTER = "---"


def list_proposal_files(proposals_dir: Path) -> list[Path]:
    """Return ``proposals/*.md`` paths in modification-time order."""
    if not proposals_dir.exists():
        return []
    files = [p for p in proposals_dir.iterdir() if p.suffix == ".md" and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def find_proposal_path(proposals_dir: Path, slug: str) -> Path | None:
    """Find a proposal by exact slug or date-prefixed markdown stem."""
    if not proposals_dir.exists():
        return None
    matches = [
        p
        for p in proposals_dir.iterdir()
        if p.suffix == ".md" and p.is_file() and p.stem.endswith(f"-{slug}")
    ]
    if matches:
        return matches[0]
    candidate = proposals_dir / f"{slug}.md"
    return candidate if candidate.is_file() else None


def proposal_path(proposals_dir: Path, proposal: ProposalModel) -> Path:
    """Build the canonical ``YYYY-MM-DD-slug.md`` path."""
    date_part = proposal.created_at.astimezone(UTC).date().isoformat()
    return proposals_dir / f"{date_part}-{proposal.slug}.md"


def load_proposal(path: Path) -> ProposalModel:
    """Read and validate one proposal frontmatter block."""
    raw, _body = load_proposal_raw(path)
    return ProposalModel.model_validate(raw)


def load_proposal_raw(path: Path) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter, body)`` from a proposal markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONTMATTER + "\n"):
        raise ValueError(f"proposal {path} is missing YAML frontmatter")
    rest = text[len(_FRONTMATTER) + 1 :]
    marker = "\n" + _FRONTMATTER
    end = rest.find(marker)
    if end < 0:
        raise ValueError(f"proposal {path} has unterminated YAML frontmatter")
    raw_text = rest[:end]
    body = rest[end + len(marker) :].lstrip("\n")
    raw = yaml.safe_load(raw_text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"proposal {path} frontmatter is not a YAML mapping")
    return raw, body


def save_proposal_raw(path: Path, data: dict[str, Any], body: str) -> None:
    """Write frontmatter and markdown body with stable YAML key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_frontmatter(data) + "\n" + body.rstrip() + "\n",
        encoding="utf-8",
    )


def save_proposal(path: Path, proposal: ProposalModel, body: str | None = None) -> None:
    """Validate and write a proposal markdown file."""
    ProposalModel.model_validate(proposal.model_dump(mode="json"))
    rendered_body = body if body is not None else render_proposal_body(proposal)
    save_proposal_raw(path, _dump_model(proposal), rendered_body)


def write_proposal(proposals_dir: Path, proposal: ProposalModel) -> Path:
    """Write proposal to its canonical path and return that path."""
    path = proposal_path(proposals_dir, proposal)
    save_proposal(path, proposal)
    return path


def render_proposal_body(proposal: ProposalModel) -> str:
    """Human-readable proposal body; the frontmatter remains the contract."""
    acceptance = "\n".join(f"- {item}" for item in proposal.acceptance)
    files = "\n".join(f"- `{item}`" for item in proposal.files_likely_touched)
    sources = "\n".join(
        f"- {source.url} ({source.trust_tier}): {source.claim}"
        for source in proposal.source_provenance
    )
    return (
        f"# {proposal.title}\n\n"
        f"## Суть\n{proposal.summary}\n\n"
        f"## Изменение\n{proposal.proposed_change}\n\n"
        f"## Почему\n{proposal.rationale}\n\n"
        f"## Проверка\n{acceptance}\n\n"
        f"## Вероятные файлы\n{files or '- уточнить после approve'}\n\n"
        f"## Риск\n{proposal.risk}\n\n"
        f"## Источники\n{sources}\n"
    )


def _dump_model(proposal: ProposalModel) -> dict[str, Any]:
    return proposal.model_dump(mode="json", exclude_none=True)


def _render_frontmatter(data: dict[str, Any]) -> str:
    return (
        _FRONTMATTER
        + "\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False).rstrip()
        + "\n"
        + _FRONTMATTER
        + "\n"
    )
