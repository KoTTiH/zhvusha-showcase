"""Priority pillar model used for topic ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[\w#+.-]{3,}", re.UNICODE)


@dataclass(frozen=True)
class Pillar:
    id: str
    name: str
    weight: float
    description: str
    success_signals: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Pillar:
        signals = tuple(str(item) for item in raw.get("success_signals", []) or [])
        keywords = tuple(str(item).lower() for item in raw.get("keywords", []) or [])
        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            weight=float(raw["weight"]),
            description=str(raw.get("description", "")),
            success_signals=signals,
            keywords=keywords,
        )


@dataclass(frozen=True)
class PillarConfig:
    version: str
    pillars: tuple[Pillar, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PillarConfig:
        pillars = tuple(Pillar.from_mapping(item) for item in raw.get("pillars", []))
        config = cls(version=str(raw.get("version", "")), pillars=pillars)
        config.validate()
        return config

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(max(0.0, pillar.weight) for pillar in self.pillars)
        if total <= 0:
            return {pillar.id: 0.0 for pillar in self.pillars}
        return {pillar.id: max(0.0, pillar.weight) / total for pillar in self.pillars}

    def validate(self) -> None:
        if not self.pillars:
            raise ValueError("pillars must not be empty")
        ids = [pillar.id for pillar in self.pillars]
        if len(ids) != len(set(ids)):
            raise ValueError("pillar ids must be unique")
        for pillar in self.pillars:
            if not pillar.name.strip():
                raise ValueError("pillar name must be non-empty")
            if pillar.weight < 0:
                raise ValueError("pillar weight must be non-negative")

    def estimate_alignment(self, text: str) -> dict[str, float]:
        """Deterministic worker-tier fallback for pillar alignment."""
        tokens = _tokens(text)
        raw_scores: dict[str, float] = {}
        for pillar in self.pillars:
            keywords = set(pillar.keywords) | _tokens(
                " ".join([pillar.name, pillar.description, *pillar.success_signals])
            )
            if not keywords:
                raw_scores[pillar.id] = 0.0
                continue
            raw_scores[pillar.id] = len(tokens & keywords) / max(1, len(keywords))
        total = sum(raw_scores.values())
        if total <= 0:
            return {pillar.id: 0.0 for pillar in self.pillars}
        return {
            pillar_id: round(score / total, 3)
            for pillar_id, score in raw_scores.items()
        }


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)}
