"""Deterministic deduplication for news source items.

The production roadmap names SemHash + MinHash. This module keeps the same
contract without adding heavyweight dependencies yet: exact URL/content
dedup first, then a conservative token similarity pass. A SemHash-backed
implementation can replace ``_text_similarity`` without changing callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.news.models import SourceItem, content_hash

_TOKEN_RE = re.compile(r"[\w#+.-]{3,}", re.UNICODE)
_TOKEN_ALIASES = {
    "кодекс": "codex",
    "кодекса": "codex",
    "опенаи": "openai",
    "опенай": "openai",
    "хуки": "hooks",
    "хуков": "hooks",
}


@dataclass(frozen=True)
class DedupDecision:
    item_id: str
    duplicate_of: str
    similarity: float
    reason: str


@dataclass(frozen=True)
class DedupResult:
    unique_items: list[SourceItem]
    duplicates: list[DedupDecision]


def deduplicate_source_items(
    items: list[SourceItem],
    *,
    same_topic_threshold: float = 0.78,
) -> DedupResult:
    """Deduplicate source items while preserving first-seen order."""
    unique: list[SourceItem] = []
    duplicates: list[DedupDecision] = []
    seen_urls: dict[str, SourceItem] = {}
    seen_hashes: dict[str, SourceItem] = {}
    token_cache: dict[int, set[str]] = {}

    for item in items:
        by_url = seen_urls.get(item.normalized_url)
        if by_url is not None:
            duplicates.append(DedupDecision(item.id, by_url.id, 1.0, "url"))
            continue

        digest = content_hash(item.title, item.body)
        by_hash = seen_hashes.get(digest)
        if by_hash is not None:
            duplicates.append(DedupDecision(item.id, by_hash.id, 1.0, "content_hash"))
            continue

        item_tokens = _item_tokens(item, token_cache)
        near_duplicate = _find_near_duplicate(
            item_tokens,
            unique,
            same_topic_threshold,
            token_cache,
        )
        if near_duplicate is not None:
            duplicate, score = near_duplicate
            duplicates.append(
                DedupDecision(item.id, duplicate.id, score, "token_similarity")
            )
            continue

        unique.append(item)
        seen_urls[item.normalized_url] = item
        seen_hashes[digest] = item

    return DedupResult(unique_items=unique, duplicates=duplicates)


def dedup_signature(item: SourceItem) -> str:
    """Stable topic-ish signature used by clustering and persistence."""
    tokens = sorted(_tokens(item.text))
    if not tokens:
        return content_hash(item.title, item.body)[:16]
    return "-".join(tokens[:8])


def _find_near_duplicate(
    item_tokens: set[str],
    candidates: list[SourceItem],
    threshold: float,
    token_cache: dict[int, set[str]],
) -> tuple[SourceItem, float] | None:
    best: tuple[SourceItem, float] | None = None
    for candidate in candidates:
        score = _token_similarity(item_tokens, _item_tokens(candidate, token_cache))
        if score < threshold:
            continue
        if best is None or score > best[1]:
            best = (candidate, score)
    return best


def _text_similarity(left: str, right: str) -> float:
    return _token_similarity(_tokens(left), _tokens(right))


def _token_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = overlap / union if union else 0.0

    # Short AI-news items often share a few high-signal entities across
    # languages; this boost is intentionally conservative and bounded.
    entity_overlap = len(_entity_tokens(left_tokens) & _entity_tokens(right_tokens))
    boost = min(0.25, entity_overlap * 0.08)
    return min(1.0, jaccard + boost)


def _item_tokens(item: SourceItem, token_cache: dict[int, set[str]]) -> set[str]:
    cache_key = id(item)
    cached = token_cache.get(cache_key)
    if cached is not None:
        return cached
    tokens = _tokens(item.text)
    token_cache[cache_key] = tokens
    return tokens


def _tokens(text: str) -> set[str]:
    result: set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.strip(".,:;()[]{}")
        if not token:
            continue
        result.add(_TOKEN_ALIASES.get(token, token))
    return result


def _entity_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if any(ch.isdigit() for ch in token)
        or token in {"ai", "api", "llm", "openai", "codex", "hooks", "github"}
        or token.startswith("gpt")
    }
