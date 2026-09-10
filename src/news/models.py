"""Domain models for external source monitoring."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

SourceTier = Literal["A", "B", "C", "D", "E"]
SourceType = Literal[
    "official_docs",
    "paper",
    "github",
    "blog",
    "secondary_press",
    "telegram",
    "youtube",
    "social",
    "other",
]


@dataclass(frozen=True)
class SourceItem:
    """Normalized article/post/release before dedup and clustering."""

    id: str
    source: str
    url: str
    title: str
    body: str
    ts: datetime
    lang: str = "en"
    source_type: SourceType = "other"
    source_tier: SourceTier = "C"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def normalized_url(self) -> str:
        return canonical_url(self.url)

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()

    def to_stream_payload(self) -> dict[str, str]:
        """Serialize to flat Redis Stream-friendly payload."""
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "body": self.body,
            "ts": self.ts.isoformat(),
            "lang": self.lang,
            "source_type": self.source_type,
            "source_tier": self.source_tier,
        }


def canonical_url(url: str) -> str:
    """Remove tracking params and normalize URL shape for duplicate checks."""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    blocked_prefixes = ("utm_",)
    blocked_exact = {"fbclid", "gclid", "yclid"}
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key not in blocked_exact and not key.startswith(blocked_prefixes)
    ]
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/",
        query=urlencode(query, doseq=True),
        fragment="",
    )
    return urlunparse(normalized)


def content_hash(title: str, body: str) -> str:
    """Stable content hash for exact duplicate detection."""
    normalized = " ".join(f"{title}\n{body}".lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_source_item_id(source: str, url: str, title: str, ts: datetime | None) -> str:
    """Generate deterministic source item ID."""
    timestamp = (ts or datetime.now(tz=UTC)).isoformat()
    raw = "\x1f".join(
        [source.strip().lower(), canonical_url(url), title.strip(), timestamp]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
