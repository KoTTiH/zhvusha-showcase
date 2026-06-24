"""Data collectors for Phase 3 knowledge sources."""

from src.collectors.arxiv import ARXIV_AI_RSS_URL, ArxivRSSCollector
from src.collectors.base import (
    BaseCollector,
    CollectionResult,
    CollectorStatus,
)
from src.collectors.github_trending import (
    OSSINSIGHT_TRENDS_URL,
    GitHubTrendingCollector,
)
from src.collectors.huggingface import HUGGINGFACE_MODELS_URL, HuggingFaceModelCollector
from src.collectors.lmarena import LM_ARENA_SNAPSHOT_URL, LMArenaSnapshotCollector
from src.collectors.rss import RSSCollector, RSSSource, parse_feed

__all__ = [
    "ARXIV_AI_RSS_URL",
    "HUGGINGFACE_MODELS_URL",
    "LM_ARENA_SNAPSHOT_URL",
    "OSSINSIGHT_TRENDS_URL",
    "ArxivRSSCollector",
    "BaseCollector",
    "CollectionResult",
    "CollectorStatus",
    "GitHubTrendingCollector",
    "HuggingFaceModelCollector",
    "LMArenaSnapshotCollector",
    "RSSCollector",
    "RSSSource",
    "parse_feed",
]
