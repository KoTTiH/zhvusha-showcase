"""RSS/Atom collectors for the news pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from src.news.models import (
    SourceItem,
    SourceTier,
    SourceType,
    make_source_item_id,
)

FetchText = Callable[[str], str]

_SOURCE_STATUSES = frozenset({"available", "degraded", "unknown", "weak"})
_UNCERTAINTY_RANK = {"low": 0, "medium": 1, "high": 2}
_IMPLEMENTATION_EVIDENCE_QUALITIES = frozenset(
    {"official", "official_docs", "paper", "technical_news", "implementation_evidence"}
)
_SKIPPED_SOURCE_STATUSES = frozenset({"degraded", "unknown"})


@dataclass(frozen=True)
class RSSSource:
    name: str
    url: str
    source_type: SourceType = "blog"
    source_tier: SourceTier = "B"
    lang: str = "en"
    source_quality: str = ""
    source_status: str = "available"
    uncertainty: str = ""
    evidence_role: str = ""
    blocker: str = ""

    def __post_init__(self) -> None:
        source_quality = _normalize_contract_value(
            self.source_quality
        ) or _default_source_quality(self.source_type)
        source_status = _normalize_source_status(self.source_status)
        blocker = self.blocker.strip()
        raw_source_status = _normalize_contract_value(self.source_status)
        if source_status == "unknown" and raw_source_status not in {"", "unknown"}:
            blocker = blocker or f"unsupported source_status={self.source_status}"

        if source_quality == "secondary_signal" or source_status == "weak":
            source_status = "weak"
            if source_quality == "source_scouting":
                source_quality = "secondary_signal"

        uncertainty = _normalize_uncertainty(self.uncertainty)
        if not uncertainty:
            uncertainty = _default_uncertainty(
                source_status=source_status,
                source_quality=source_quality,
            )
        if source_status in {"degraded", "unknown"}:
            uncertainty = _ensure_min_uncertainty(uncertainty, "high")
        elif source_status == "weak" or source_quality == "secondary_signal":
            uncertainty = _ensure_min_uncertainty(uncertainty, "medium")

        evidence_role = _normalize_contract_value(self.evidence_role)
        if not evidence_role:
            evidence_role = _default_evidence_role(
                source_status=source_status,
                source_quality=source_quality,
            )
        evidence_role = _enforce_evidence_role(
            source_status=source_status,
            source_quality=source_quality,
            evidence_role=evidence_role,
        )

        object.__setattr__(self, "source_quality", source_quality)
        object.__setattr__(self, "source_status", source_status)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "evidence_role", evidence_role)
        object.__setattr__(self, "blocker", blocker)


@dataclass(frozen=True)
class RSSSourceCollectionReport:
    name: str
    url: str
    source_status: str
    source_quality: str
    uncertainty: str
    evidence_role: str
    item_count: int
    blocker: str = ""


class RSSCollector:
    """Collect normalized ``SourceItem`` objects from RSS/Atom feeds."""

    def __init__(
        self,
        sources: list[RSSSource],
        *,
        fetch_text: FetchText | None = None,
    ) -> None:
        self._sources = sources
        self._fetch_text = fetch_text or _fetch_text
        self._fetch_in_thread = fetch_text is None
        self._last_collection_report: tuple[RSSSourceCollectionReport, ...] = ()

    @property
    def last_collection_report(self) -> tuple[RSSSourceCollectionReport, ...]:
        return self._last_collection_report

    async def collect(self) -> list[SourceItem]:
        items: list[SourceItem] = []
        reports: list[RSSSourceCollectionReport] = []
        for source in self._sources:
            if _should_skip_source(source):
                reports.append(
                    _collection_report(
                        source,
                        item_count=0,
                        blocker=source.blocker
                        or f"source_status={source.source_status} is not approved",
                    )
                )
                continue
            try:
                if self._fetch_in_thread:
                    xml = await asyncio.to_thread(self._fetch_text, source.url)
                else:
                    xml = self._fetch_text(source.url)
                parsed_items = parse_feed(xml, source)
            except Exception as exc:
                if not _can_report_failure(source):
                    self._last_collection_report = tuple(reports)
                    raise
                reports.append(
                    _collection_report(
                        source,
                        item_count=0,
                        source_status=_failure_status(source),
                        blocker=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            items.extend(parsed_items)
            reports.append(_collection_report(source, item_count=len(parsed_items)))
        self._last_collection_report = tuple(reports)
        return items


def parse_feed(xml: str, source: RSSSource) -> list[SourceItem]:
    """Parse RSS 2.0 or Atom XML into normalized source items."""
    _reject_unsafe_xml(xml)
    root = ElementTree.fromstring(xml)  # noqa: S314
    if _strip_ns(root.tag) == "feed":
        return _parse_atom(root, source)
    return _parse_rss(root, source)


def _parse_rss(root: ElementTree.Element, source: RSSSource) -> list[SourceItem]:
    items: list[SourceItem] = []
    for node in root.findall("./channel/item"):
        title = _child_text(node, "title")
        url = _child_text(node, "link")
        body = _child_text(node, "description")
        ts = _parse_dt(_child_text(node, "pubDate"))
        if not title or not url:
            continue
        items.append(_make_item(source, title=title, url=url, body=body, ts=ts))
    return items


def _parse_atom(root: ElementTree.Element, source: RSSSource) -> list[SourceItem]:
    items: list[SourceItem] = []
    for node in root.findall("./{*}entry"):
        title = _child_text(node, "title")
        url = _atom_link(node)
        body = _child_text(node, "summary") or _child_text(node, "content")
        ts = _parse_dt(_child_text(node, "updated") or _child_text(node, "published"))
        if not title or not url:
            continue
        items.append(_make_item(source, title=title, url=url, body=body, ts=ts))
    return items


def _make_item(
    source: RSSSource,
    *,
    title: str,
    url: str,
    body: str,
    ts: datetime,
) -> SourceItem:
    return SourceItem(
        id=make_source_item_id(source.name, url, title, ts),
        source=source.name,
        url=url,
        title=title.strip(),
        body=body.strip(),
        ts=ts,
        lang=source.lang,
        source_type=source.source_type,
        source_tier=source.source_tier,
        metadata={
            "feed_url": source.url,
            "source": source.name,
            "published_at": ts.isoformat(),
            "summary": body.strip(),
            "source_quality": source.source_quality,
            "source_status": source.source_status,
            "uncertainty": source.uncertainty,
            "evidence_role": source.evidence_role,
        },
    )


def _collection_report(
    source: RSSSource,
    *,
    item_count: int,
    source_status: str | None = None,
    blocker: str = "",
) -> RSSSourceCollectionReport:
    status = source_status or source.source_status
    evidence_role = _enforce_evidence_role(
        source_status=status,
        source_quality=source.source_quality,
        evidence_role=source.evidence_role,
    )
    uncertainty = source.uncertainty
    if status in {"degraded", "unknown"}:
        uncertainty = _ensure_min_uncertainty(uncertainty, "high")
    elif status == "weak" or source.source_quality == "secondary_signal":
        uncertainty = _ensure_min_uncertainty(uncertainty, "medium")
    return RSSSourceCollectionReport(
        name=source.name,
        url=source.url,
        source_status=status,
        source_quality=source.source_quality,
        uncertainty=uncertainty,
        evidence_role=evidence_role,
        item_count=item_count,
        blocker=blocker or source.blocker,
    )


def _should_skip_source(source: RSSSource) -> bool:
    return source.source_status in _SKIPPED_SOURCE_STATUSES


def _can_report_failure(source: RSSSource) -> bool:
    return (
        source.source_status != "available"
        or source.source_quality in {"secondary_signal", "unknown"}
        or bool(source.blocker)
    )


def _failure_status(source: RSSSource) -> str:
    if source.source_status in {"unknown", "weak"}:
        return source.source_status
    return "degraded"


def _normalize_contract_value(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _normalize_source_status(value: str) -> str:
    status = _normalize_contract_value(value or "available")
    if status in _SOURCE_STATUSES:
        return status
    return "unknown"


def _normalize_uncertainty(value: str) -> str:
    uncertainty = _normalize_contract_value(value)
    if uncertainty in _UNCERTAINTY_RANK:
        return uncertainty
    return ""


def _default_source_quality(source_type: SourceType) -> str:
    if source_type == "official_docs":
        return "official"
    if source_type == "paper":
        return "paper"
    if source_type == "secondary_press":
        return "secondary_signal"
    return "source_scouting"


def _default_uncertainty(*, source_status: str, source_quality: str) -> str:
    if source_status in {"degraded", "unknown"}:
        return "high"
    if source_status == "weak" or source_quality == "secondary_signal":
        return "medium"
    if source_quality in _IMPLEMENTATION_EVIDENCE_QUALITIES:
        return "low"
    return "medium"


def _ensure_min_uncertainty(value: str, minimum: str) -> str:
    if _UNCERTAINTY_RANK[value] >= _UNCERTAINTY_RANK[minimum]:
        return value
    return minimum


def _default_evidence_role(*, source_status: str, source_quality: str) -> str:
    if source_status in {"degraded", "unknown"}:
        return "blocked"
    if source_status == "weak" or source_quality == "secondary_signal":
        return "secondary_signal"
    if source_quality in _IMPLEMENTATION_EVIDENCE_QUALITIES:
        return "implementation_evidence"
    return "source_scouting"


def _enforce_evidence_role(
    *,
    source_status: str,
    source_quality: str,
    evidence_role: str,
) -> str:
    if source_status in {"degraded", "unknown"}:
        return "blocked"
    if source_status == "weak" or source_quality == "secondary_signal":
        return "secondary_signal"
    if (
        evidence_role == "implementation_evidence"
        and source_quality not in _IMPLEMENTATION_EVIDENCE_QUALITIES
    ):
        return "source_scouting"
    return evidence_role


def _child_text(node: ElementTree.Element, child_name: str) -> str:
    for child in list(node):
        if _strip_ns(child.tag) == child_name:
            return "".join(child.itertext()).strip()
    return ""


def _atom_link(node: ElementTree.Element) -> str:
    for child in list(node):
        if _strip_ns(child.tag) == "link":
            href = child.attrib.get("href", "").strip()
            if href:
                return href
    return ""


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fetch_text(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported RSS URL scheme: {parsed.scheme}")
    request = Request(url, headers={"User-Agent": "zhvusha-news-monitor/1.0"})  # noqa: S310
    with urlopen(request, timeout=20) as response:  # noqa: S310
        data = response.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _reject_unsafe_xml(xml: str) -> None:
    lowered = xml[:500].lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError("RSS XML with DTD/entities is not accepted")
