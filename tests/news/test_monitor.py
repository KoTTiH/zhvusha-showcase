"""Active news monitor loop tests."""

from __future__ import annotations

from datetime import UTC, datetime

from src.news.models import SourceItem


class _Collector:
    def __init__(self, items: list[SourceItem]) -> None:
        self.items = items
        self.calls = 0

    async def collect(self) -> list[SourceItem]:
        self.calls += 1
        return self.items


class _Store:
    def __init__(self) -> None:
        self.item_batches: list[list[SourceItem]] = []
        self.cluster_batches: list[object] = []

    async def upsert_items(self, items: list[SourceItem]) -> int:
        self.item_batches.append(items)
        return len(items)

    async def upsert_clusters(self, clusters: list[object]) -> int:
        self.cluster_batches.append(clusters)
        return len(clusters)


class _Redis:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, str]]] = []

    async def xadd(
        self,
        stream_name: str,
        payload: dict[str, str],
        *,
        maxlen: int,
        approximate: bool,
    ) -> str:
        del maxlen, approximate
        self.entries.append((stream_name, payload))
        return "1-0"


def _item(item_id: str, title: str) -> SourceItem:
    return SourceItem(
        id=item_id,
        source="test",
        url=f"https://example.com/{item_id}",
        title=title,
        body=f"Codex self-coding archive gates {item_id}.",
        ts=datetime(2026, 5, 7, tzinfo=UTC),
        source_type="official_docs",
        source_tier="A",
    )


async def test_news_monitor_collects_processes_persists_and_publishes() -> None:
    from src.news.monitor import NewsMonitor

    collector = _Collector(
        [
            _item("a", "Codex archive gates"),
            _item("b", "Kwork client signal"),
        ]
    )
    store = _Store()
    redis = _Redis()
    monitor = NewsMonitor(
        collectors=[collector],
        store=store,  # type: ignore[arg-type]
        redis=redis,
        stream_name="news:raw:test",
    )

    result = await monitor.poll_once()

    assert collector.calls == 1
    assert result.collected_count == 2
    assert result.unique_count == 2
    assert result.cluster_count >= 1
    assert len(store.item_batches[0]) == 2
    assert redis.entries[0][0] == "news:raw:test"


async def test_news_processing_offloads_cpu_bound_steps(monkeypatch) -> None:
    from src.news import pipeline

    calls: list[str] = []

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        calls.append(getattr(func, "__name__", "unknown"))
        return func(*args, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(pipeline.asyncio, "to_thread", fake_to_thread)

    await pipeline.process_source_items([_item("a", "Codex archive gates")])

    assert calls == ["deduplicate_source_items", "cluster_source_items"]


def test_build_default_news_collectors_uses_arxiv_and_extra_rss_urls() -> None:
    from src.news.monitor import build_default_news_collectors

    collectors = build_default_news_collectors(
        arxiv_url="https://rss.arxiv.org/rss/cs.AI",
        rss_urls="openai=https://example.com/openai.xml,https://example.com/blog.xml",
    )

    assert len(collectors) == 2


# fmt: off
async def test_rss_news_sources_preserve_source_status_quality_and_skip_degraded_feeds() -> None:
# fmt: on
    from src.collectors.rss import RSSCollector, RSSSource
    from src.news.monitor import NewsMonitor

    feeds = {
        "https://openai.com/news/rss.xml": """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>OpenAI Codex release notes</title>
              <link>https://openai.com/news/codex</link>
              <description>Official Codex release details.</description>
              <pubDate>Thu, 28 May 2026 08:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """,
        "https://rss.arxiv.org/rss/cs.AI": """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>Agentic Source Evaluation</title>
              <link>https://arxiv.org/abs/2605.12345?utm_source=rss</link>
              <description>Paper summary from arXiv.</description>
              <pubDate>Thu, 28 May 2026 09:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """,
        "https://www.infoq.com/ai/news/rss/": """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>InfoQ AI engineering report</title>
              <link>https://www.infoq.com/news/2026/05/ai-engineering/</link>
              <description>Technical reporting with named sources.</description>
              <pubDate>Thu, 28 May 2026 10:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """,
        "https://venturebeat.com/category/ai/feed/": """<?xml version="1.0"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>AI market trend roundup</title>
              <link>https://venturebeat.com/ai/trend-roundup/</link>
              <description>Weak trend signal without primary provenance.</description>
              <pubDate>Thu, 28 May 2026 11:00:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """,
    }
    requested_urls: list[str] = []

    def fetch_text(url: str) -> str:
        requested_urls.append(url)
        return feeds[url]

    collector = RSSCollector(
        [
            RSSSource(
                name="openai-official",
                url="https://openai.com/news/rss.xml",
                source_type="official_docs",
                source_tier="A",
                source_quality="official",
                source_status="available",
                uncertainty="low",
            ),
            RSSSource(
                name="arxiv-ai",
                url="https://rss.arxiv.org/rss/cs.AI",
                source_type="paper",
                source_tier="B",
                source_quality="paper",
                source_status="available",
                uncertainty="low",
            ),
            RSSSource(
                name="infoq-ai",
                url="https://www.infoq.com/ai/news/rss/",
                source_type="secondary_press",
                source_tier="B",
                source_quality="technical_news",
                source_status="available",
                uncertainty="low",
            ),
            RSSSource(
                name="anthropic-guessed",
                url="https://www.anthropic.com/news/rss.xml",
                source_quality="unknown",
                source_status="degraded",
                uncertainty="high",
                blocker="guessed feed endpoint from scouting failed",
            ),
            RSSSource(
                name="venturebeat-ai",
                url="https://venturebeat.com/category/ai/feed/",
                source_type="secondary_press",
                source_tier="C",
                source_quality="secondary_signal",
                source_status="weak",
                uncertainty="medium",
            ),
        ],
        fetch_text=fetch_text,
    )
    store = _Store()
    monitor = NewsMonitor(
        collectors=[collector],
        store=store,  # type: ignore[arg-type]
    )

    result = await monitor.poll_once()

    assert "https://www.anthropic.com/news/rss.xml" not in requested_urls
    assert result.collected_count == 4
    assert result.unique_count == 4

    items_by_source = {item.source: item for item in store.item_batches[0]}
    assert set(items_by_source) == {
        "openai-official",
        "arxiv-ai",
        "infoq-ai",
        "venturebeat-ai",
    }

    openai_item = items_by_source["openai-official"]
    assert openai_item.title == "OpenAI Codex release notes"
    assert openai_item.url == "https://openai.com/news/codex"
    assert openai_item.ts.isoformat() == "2026-05-28T08:00:00+00:00"
    assert openai_item.body == "Official Codex release details."
    assert openai_item.metadata["source"] == "openai-official"
    assert openai_item.metadata["summary"] == "Official Codex release details."
    assert openai_item.metadata["published_at"] == "2026-05-28T08:00:00+00:00"
    assert openai_item.metadata["source_quality"] == "official"
    assert openai_item.metadata["source_status"] == "available"

    arxiv_item = items_by_source["arxiv-ai"]
    assert arxiv_item.normalized_url == "https://arxiv.org/abs/2605.12345"
    assert arxiv_item.metadata["source_quality"] == "paper"
    assert arxiv_item.metadata["source_status"] == "available"

    infoq_item = items_by_source["infoq-ai"]
    assert infoq_item.metadata["source_quality"] == "technical_news"
    assert infoq_item.metadata["source_status"] == "available"

    weak_item = items_by_source["venturebeat-ai"]
    assert weak_item.metadata["source_quality"] == "secondary_signal"
    assert weak_item.metadata["source_status"] == "weak"
    assert weak_item.metadata["evidence_role"] == "secondary_signal"
    assert weak_item.metadata["evidence_role"] != "implementation_evidence"
    assert {"low": 0, "medium": 1, "high": 2}[weak_item.metadata["uncertainty"]] >= 1

    reports = {report.name: report for report in result.source_reports}
    assert reports["anthropic-guessed"].source_status == "degraded"
    assert reports["anthropic-guessed"].source_quality == "unknown"
    assert reports["anthropic-guessed"].evidence_role == "blocked"
    assert reports["anthropic-guessed"].item_count == 0
    assert "guessed feed endpoint" in reports["anthropic-guessed"].blocker
    assert reports["venturebeat-ai"].source_status == "weak"
    assert reports["venturebeat-ai"].source_quality == "secondary_signal"
    assert reports["venturebeat-ai"].evidence_role == "secondary_signal"
    assert {"low": 0, "medium": 1, "high": 2}[
        reports["venturebeat-ai"].uncertainty
    ] >= 1
