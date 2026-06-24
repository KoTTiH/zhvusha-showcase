"""ArXiv RSS source definitions."""

from __future__ import annotations

from src.collectors.rss import FetchText, RSSCollector, RSSSource

ARXIV_AI_RSS_URL = "https://rss.arxiv.org/rss/cs.AI+cs.CL+cs.SE"


class ArxivRSSCollector(RSSCollector):
    """Collector for the AI/CL/SE ArXiv RSS feed."""

    def __init__(
        self,
        *,
        feed_url: str = ARXIV_AI_RSS_URL,
        fetch_text: FetchText | None = None,
    ) -> None:
        super().__init__(
            [
                RSSSource(
                    name="arxiv-ai-cl-se",
                    url=feed_url,
                    source_type="paper",
                    source_tier="B",
                    lang="en",
                )
            ],
            fetch_text=fetch_text,
        )
