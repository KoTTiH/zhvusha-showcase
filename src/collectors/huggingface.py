"""HuggingFace model source collector for the news pipeline."""

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

HUGGINGFACE_MODELS_URL = (
    "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50"
)


@dataclass(frozen=True)
class HuggingFaceModel:
    model_id: str
    downloads: int = 0
    likes: int = 0
    last_modified: datetime | None = None
    tags: tuple[str, ...] = ()


class HuggingFaceModelCollector:
    """Collect popular/recent HuggingFace model records."""

    def __init__(
        self,
        *,
        url: str = HUGGINGFACE_MODELS_URL,
        fetch_json: FetchJSON | None = None,
    ) -> None:
        self._url = url
        self._fetch_json = fetch_json or _fetch_json

    async def collect(self) -> list[SourceItem]:
        payload = await asyncio.to_thread(self._fetch_json, self._url)
        return models_to_source_items(parse_huggingface_models(payload))


def parse_huggingface_models(payload: Any) -> list[HuggingFaceModel]:
    if isinstance(payload, dict):
        rows = (
            payload.get("models") or payload.get("data") or payload.get("items") or []
        )
    else:
        rows = payload
    if not isinstance(rows, list):
        return []

    models: list[HuggingFaceModel] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("modelId") or row.get("id") or "").strip()
        if not model_id:
            continue
        tags = tuple(str(tag) for tag in row.get("tags", []) if isinstance(tag, str))
        models.append(
            HuggingFaceModel(
                model_id=model_id,
                downloads=int(row.get("downloads") or 0),
                likes=int(row.get("likes") or 0),
                last_modified=_parse_dt(str(row.get("lastModified") or "")),
                tags=tags,
            )
        )
    return models


def models_to_source_items(
    models: list[HuggingFaceModel],
    *,
    ts: datetime | None = None,
) -> list[SourceItem]:
    collected_at = ts or datetime.now(tz=UTC)
    items: list[SourceItem] = []
    for model in models:
        modified = model.last_modified or collected_at
        url = f"https://huggingface.co/{model.model_id}"
        title = f"HuggingFace model: {model.model_id}"
        body = (
            f"Downloads: {model.downloads}\n"
            f"Likes: {model.likes}\n"
            f"Tags: {', '.join(model.tags) if model.tags else 'none'}"
        )
        items.append(
            SourceItem(
                id=make_source_item_id("huggingface-models", url, title, modified),
                source="huggingface-models",
                url=url,
                title=title,
                body=body,
                ts=modified,
                lang="en",
                source_type="github",
                source_tier="A",
                metadata={
                    "downloads": str(model.downloads),
                    "likes": str(model.likes),
                    "tags": ",".join(model.tags),
                    "collected_at": collected_at.isoformat(),
                },
            )
        )
    return items


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fetch_json(url: str) -> Any:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme}")
    request = Request(url, headers={"User-Agent": "zhvusha-news-monitor/1.0"})  # noqa: S310
    with urlopen(request, timeout=20) as response:  # noqa: S310
        data = response.read()
    return json.loads(data.decode("utf-8", errors="replace"))
