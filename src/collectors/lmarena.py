"""LM Arena snapshot collector for the news pipeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.news.models import SourceItem, make_source_item_id

FetchJSON = Callable[[str], Any]

LM_ARENA_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/oolong-tea-2026/arena-ai-leaderboards/main/"
    "leaderboards.json"
)


@dataclass(frozen=True)
class LMArenaModel:
    model: str
    rank: int | None = None
    score: float | None = None
    organization: str = ""


class LMArenaSnapshotCollector:
    """Collect model leaderboard snapshots as news radar items."""

    def __init__(
        self,
        *,
        url: str = LM_ARENA_SNAPSHOT_URL,
        fetch_json: FetchJSON | None = None,
    ) -> None:
        self._url = url
        self._fetch_json = fetch_json or _fetch_json

    async def collect(self) -> list[SourceItem]:
        payload = await asyncio.to_thread(self._fetch_json, self._url)
        return arena_models_to_source_items(
            parse_lmarena_models(payload), source_url=self._url
        )


def parse_lmarena_models(payload: Any) -> list[LMArenaModel]:
    rows = _rows_from_payload(payload)
    models: list[LMArenaModel] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or row.get("name") or row.get("model_name") or "")
        if not model:
            continue
        rank = row.get("rank")
        score = row.get("score") or row.get("arena_score") or row.get("elo")
        models.append(
            LMArenaModel(
                model=model,
                rank=int(rank) if rank is not None else idx,
                score=float(score) if score is not None else None,
                organization=str(row.get("organization") or row.get("org") or ""),
            )
        )
    return models


def arena_models_to_source_items(
    models: list[LMArenaModel],
    *,
    source_url: str = LM_ARENA_SNAPSHOT_URL,
    ts: datetime | None = None,
) -> list[SourceItem]:
    collected_at = ts or datetime.now(tz=UTC)
    items: list[SourceItem] = []
    for model in models:
        rank = f"#{model.rank}" if model.rank is not None else "unranked"
        title = f"LM Arena: {model.model} {rank}"
        body = f"Model: {model.model}\nRank: {rank}"
        if model.score is not None:
            body += f"\nScore: {model.score}"
        if model.organization:
            body += f"\nOrganization: {model.organization}"
        items.append(
            SourceItem(
                id=make_source_item_id("lm-arena", source_url, title, collected_at),
                source="lm-arena",
                url=source_url,
                title=title,
                body=body,
                ts=collected_at,
                lang="en",
                source_type="other",
                source_tier="A",
                metadata={
                    "model": model.model,
                    "rank": str(model.rank or ""),
                    "score": str(model.score or ""),
                    "organization": model.organization,
                },
            )
        )
    return items


def _rows_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("leaderboard", "models", "data", "items", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _fetch_json(url: str) -> Any:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme}")
    request = Request(url, headers={"User-Agent": "zhvusha-news-monitor/1.0"})  # noqa: S310
    with urlopen(request, timeout=20) as response:  # noqa: S310
        data = response.read()
    return json.loads(data.decode("utf-8", errors="replace"))
