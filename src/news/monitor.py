"""Active source polling for the news/topic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from src.collectors.arxiv import ArxivRSSCollector
from src.collectors.rss import RSSCollector, RSSSource, RSSSourceCollectionReport
from src.news.pipeline import NEWS_RAW_STREAM, process_source_items

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.news.models import SourceItem, SourceTier, SourceType
    from src.news.store import NewsStore
    from src.pillars import PillarConfig


class SourceCollector(Protocol):
    async def collect(self) -> list[SourceItem]: ...


@dataclass(frozen=True)
class NewsMonitorResult:
    collected_count: int
    unique_count: int
    duplicate_count: int
    cluster_count: int
    source_reports: tuple[RSSSourceCollectionReport, ...] = ()


class NewsMonitor:
    """Poll configured source collectors and feed the news pipeline."""

    def __init__(
        self,
        *,
        collectors: list[SourceCollector],
        store: NewsStore | None = None,
        pillars: PillarConfig | None = None,
        redis: object | None = None,
        stream_name: str = NEWS_RAW_STREAM,
    ) -> None:
        self._collectors = collectors
        self._store = store
        self._pillars = pillars
        self._redis = redis
        self._stream_name = stream_name

    async def poll_once(self) -> NewsMonitorResult:
        items: list[SourceItem] = []
        source_reports: list[RSSSourceCollectionReport] = []
        for collector in self._collectors:
            items.extend(await collector.collect())
            source_reports.extend(_collector_reports(collector))
        deduped, clusters = await process_source_items(
            items,
            store=self._store,
            pillars=self._pillars,
            redis=self._redis,
            stream_name=self._stream_name,
        )
        return NewsMonitorResult(
            collected_count=len(items),
            unique_count=len(deduped.unique_items),
            duplicate_count=len(deduped.duplicates),
            cluster_count=len(clusters),
            source_reports=tuple(source_reports),
        )


def build_default_news_collectors(
    *, arxiv_url: str, rss_urls: str
) -> list[SourceCollector]:
    """Build the enabled collector set from settings strings."""
    collectors: list[SourceCollector] = [ArxivRSSCollector(feed_url=arxiv_url)]
    sources: list[RSSSource] = []
    for token in _split_csv(rss_urls):
        source = _rss_source_from_token(token)
        if source is not None:
            sources.append(source)
    if sources:
        collectors.append(RSSCollector(sources))
    return collectors


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _rss_source_from_token(token: str) -> RSSSource | None:
    if not token:
        return None
    base_token, metadata = _split_source_metadata(token)
    if "=" in base_token:
        name, url = base_token.split("=", maxsplit=1)
        name = name.strip()
        url = url.strip()
    else:
        url = base_token.strip()
        name = _source_name_from_url(url)
    name = metadata.get("name", name).strip()
    url = metadata.get("url", url).strip()
    if not name or not url:
        return None
    return RSSSource(
        name=name,
        url=url,
        source_type=_source_type_from_metadata(metadata.get("source_type"), "blog"),
        source_tier=_source_tier_from_metadata(metadata.get("source_tier"), "B"),
        lang=metadata.get("lang", "en"),
        source_quality=_metadata_value(metadata, "source_quality", "quality"),
        source_status=_metadata_value(metadata, "source_status", "status")
        or "available",
        uncertainty=metadata.get("uncertainty", ""),
        evidence_role=metadata.get("evidence_role", ""),
        blocker=metadata.get("blocker", ""),
    )


def _source_name_from_url(url: str) -> str:
    cleaned = url.removeprefix("https://").removeprefix("http://")
    cleaned = cleaned.split("/", maxsplit=1)[0]
    return cleaned.replace(".", "-") or "rss-source"


def _collector_reports(
    collector: SourceCollector,
) -> tuple[RSSSourceCollectionReport, ...]:
    reports = getattr(collector, "last_collection_report", None)
    if reports is None:
        return ()
    return tuple(cast("Iterable[RSSSourceCollectionReport]", reports))


def _split_source_metadata(token: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in token.split("|") if part.strip()]
    if not parts:
        return "", {}
    metadata: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", maxsplit=1)
        metadata[_metadata_key(key)] = value.strip()
    return parts[0], metadata


def _metadata_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _metadata_value(metadata: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value:
            return value
    return ""


def _source_type_from_metadata(value: str | None, default: SourceType) -> SourceType:
    allowed = {
        "official_docs",
        "paper",
        "github",
        "blog",
        "secondary_press",
        "telegram",
        "youtube",
        "social",
        "other",
    }
    normalized = _metadata_key(value or "")
    if normalized in allowed:
        return cast("SourceType", normalized)
    return default


def _source_tier_from_metadata(value: str | None, default: SourceTier) -> SourceTier:
    normalized = (value or "").strip().upper()
    if normalized in {"A", "B", "C", "D", "E"}:
        return cast("SourceTier", normalized)
    return default
