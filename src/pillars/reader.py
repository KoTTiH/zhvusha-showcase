"""Read-only loader for Nikita-owned priority pillars."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pillars.models import PillarConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_pillars(path: Path) -> PillarConfig:
    """Load pillars from YAML or a Markdown file containing YAML."""
    text = path.expanduser().read_text(encoding="utf-8")
    raw = _safe_load_yaml(_extract_yaml(text)) or {}
    if not isinstance(raw, dict):
        raise ValueError("pillars file must contain a YAML mapping")
    return PillarConfig.from_mapping(raw)


def render_default_pillars() -> str:
    """Default template from the architecture plan; Никита owns real values."""
    return """version: 2026-05-07
pillars:
  - id: self_improvement
    name: "Самосовершенствование Жвуши"
    weight: 0.40
    description: "Новые skills, лучшая память, чище архитектура."
    keywords: ["codex", "самокодинг", "skill", "архитектура", "память"]
    success_signals:
      - new skill shipped
      - capability module refactor with green tests
  - id: personality
    name: "Прокачка характера Жвуши"
    weight: 0.30
    description: "Характер, тон, отношения, diary, self-reflection."
    keywords: ["личность", "тон", "дневник", "память", "общение"]
    success_signals:
      - meaningful diary entries
      - chat quality increase
  - id: money
    name: "Заработок денег для Никиты"
    weight: 0.30
    description: "Kwork, контент-маркетинг, посты, клиенты."
    keywords: ["kwork", "пост", "клиент", "деньги", "канал"]
    success_signals:
      - channel post engagement
      - prospective client identified
"""


def _extract_yaml(text: str) -> str:
    stripped = text.strip()
    front_matter = _extract_front_matter(stripped)
    if front_matter is not None:
        return front_matter
    fenced = _extract_fenced_yaml(stripped)
    if fenced is not None:
        return fenced
    return text


def _extract_front_matter(text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    lines = text.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in {"---", "..."}:
            return "\n".join(lines[1:index])
    return None


def _extract_fenced_yaml(text: str) -> str | None:
    lines = text.splitlines()
    for start, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        language = stripped.removeprefix("```").strip().lower()
        if language not in {"", "yaml", "yml"}:
            continue
        for end, closing in enumerate(lines[start + 1 :], start=start + 1):
            if closing.strip() == "```":
                return "\n".join(lines[start + 1 : end])
    return None


def _safe_load_yaml(text: str) -> object:
    import yaml

    return yaml.safe_load(text)
