"""RSS/ArXiv normalization tests for Phase 16."""

from __future__ import annotations

from src.collectors.arxiv import ARXIV_AI_RSS_URL, ArxivRSSCollector
from src.collectors.rss import RSSSource, parse_feed


def test_arxiv_rss_normalizes_to_source_item() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>cs.AI updates</title>
        <item>
          <title>Codex Hooks for Agent Safety</title>
          <link>https://arxiv.org/abs/2605.00001?utm_source=test</link>
          <description>We study deterministic hooks for coding agents.</description>
          <pubDate>Thu, 07 May 2026 04:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    items = parse_feed(
        feed,
        RSSSource(
            name="arxiv-ai-cl-se",
            url=ARXIV_AI_RSS_URL,
            source_type="paper",
            source_tier="B",
        ),
    )

    assert len(items) == 1
    item = items[0]
    assert item.source == "arxiv-ai-cl-se"
    assert item.source_type == "paper"
    assert item.source_tier == "B"
    assert item.title == "Codex Hooks for Agent Safety"
    assert item.normalized_url == "https://arxiv.org/abs/2605.00001"
    assert item.ts.isoformat() == "2026-05-07T04:00:00+00:00"


async def test_arxiv_collector_accepts_mock_feed() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Agentic Evaluation for Code Models</title>
          <link>https://arxiv.org/abs/2605.00002?utm_campaign=feed</link>
          <description>Benchmarks for AI coding agents.</description>
          <pubDate>Thu, 07 May 2026 06:30:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    requested_urls: list[str] = []

    def fetch_text(url: str) -> str:
        requested_urls.append(url)
        return feed

    collector = ArxivRSSCollector(fetch_text=fetch_text)

    items = await collector.collect()

    assert requested_urls == [ARXIV_AI_RSS_URL]
    assert len(items) == 1
    assert items[0].source == "arxiv-ai-cl-se"
    assert items[0].source_type == "paper"
    assert items[0].normalized_url == "https://arxiv.org/abs/2605.00002"


def test_atom_feed_normalizes_to_source_item() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>OpenAI Codex changelog</title>
        <link href="https://developers.openai.com/codex/hooks" />
        <summary>New hook lifecycle events.</summary>
        <updated>2026-05-07T05:00:00Z</updated>
      </entry>
    </feed>
    """

    items = parse_feed(
        feed,
        RSSSource(
            name="openai-docs",
            url="https://developers.openai.com/news/rss.xml",
            source_type="official_docs",
            source_tier="A",
        ),
    )

    assert len(items) == 1
    assert items[0].source_type == "official_docs"
    assert items[0].url == "https://developers.openai.com/codex/hooks"
