"""Pillars contract tests for Phase 21."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.pillars import PillarConfig, load_pillars, render_default_pillars


def test_load_default_pillars_template(tmp_path: Path) -> None:
    path = tmp_path / "pillars.md"
    path.write_text(render_default_pillars(), encoding="utf-8")

    config = load_pillars(path)

    assert len(config.pillars) == 3
    assert round(sum(config.normalized_weights.values()), 6) == 1.0
    assert "self_improvement" in config.normalized_weights


def test_load_pillars_from_markdown_fence(tmp_path: Path) -> None:
    path = tmp_path / "pillars.md"
    path.write_text(
        """# Никитины столпы

```yaml
version: test
pillars:
  - id: self
    name: "Саморазвитие"
    weight: 1
    keywords: ["codex"]
```
""",
        encoding="utf-8",
    )

    config = load_pillars(path)

    assert config.version == "test"
    assert config.pillars[0].id == "self"


def test_duplicate_pillar_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        PillarConfig.from_mapping(
            {
                "version": "test",
                "pillars": [
                    {"id": "x", "name": "A", "weight": 1},
                    {"id": "x", "name": "B", "weight": 1},
                ],
            }
        )


def test_alignment_uses_nikita_defined_keywords() -> None:
    config = PillarConfig.from_mapping(
        {
            "version": "test",
            "pillars": [
                {
                    "id": "money",
                    "name": "Деньги",
                    "weight": 1,
                    "keywords": ["kwork", "клиент"],
                },
                {
                    "id": "personality",
                    "name": "Личность",
                    "weight": 1,
                    "keywords": ["тон", "дневник"],
                },
            ],
        }
    )

    alignment = config.estimate_alignment("Kwork клиент просит Telegram bot")

    assert alignment["money"] == 1.0
    assert alignment["personality"] == 0.0
