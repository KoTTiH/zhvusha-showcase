"""GitHub trending source collector for the news pipeline."""

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

OSSINSIGHT_TRENDS_URL = "https://api.ossinsight.io/v1/trends/repos/"


@dataclass(frozen=True)
class GitHubTrendingRepo:
    full_name: str
    description: str
    url: str
    stars: int = 0
    language: str = ""


class GitHubTrendingCollector:
    """Collect trending GitHub repositories through OSSInsight-compatible JSON."""

    def __init__(
        self,
        *,
        url: str = OSSINSIGHT_TRENDS_URL,
        fetch_json: FetchJSON | None = None,
    ) -> None:
        self._url = url
        self._fetch_json = fetch_json or _fetch_json

    async def collect(self) -> list[SourceItem]:
        payload = await asyncio.to_thread(self._fetch_json, self._url)
        return repos_to_source_items(parse_ossinsight_repos(payload))


def parse_ossinsight_repos(payload: Any) -> list[GitHubTrendingRepo]:
    """Parse OSSInsight or GitHub-like repo payloads into stable repo records."""
    rows = _rows_from_payload(payload)
    repos: list[GitHubTrendingRepo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        full_name = str(
            row.get("full_name")
            or row.get("repo_name")
            or row.get("name")
            or row.get("repository")
            or ""
        ).strip()
        if not full_name:
            owner = str(row.get("owner", "")).strip()
            repo = str(row.get("repo", "")).strip()
            full_name = f"{owner}/{repo}" if owner and repo else ""
        if not full_name:
            continue
        url = str(
            row.get("url") or row.get("html_url") or f"https://github.com/{full_name}"
        )
        repos.append(
            GitHubTrendingRepo(
                full_name=full_name,
                description=str(row.get("description") or ""),
                url=url,
                stars=int(row.get("stars") or row.get("stargazers_count") or 0),
                language=str(row.get("language") or ""),
            )
        )
    return repos


def repos_to_source_items(
    repos: list[GitHubTrendingRepo],
    *,
    ts: datetime | None = None,
) -> list[SourceItem]:
    collected_at = ts or datetime.now(tz=UTC)
    items: list[SourceItem] = []
    for repo in repos:
        title = f"GitHub trending: {repo.full_name}"
        body = repo.description or "Trending GitHub repository."
        if repo.stars:
            body += f"\nStars: {repo.stars}"
        if repo.language:
            body += f"\nLanguage: {repo.language}"
        items.append(
            SourceItem(
                id=make_source_item_id(
                    "github-trending", repo.url, title, collected_at
                ),
                source="github-trending",
                url=repo.url,
                title=title,
                body=body,
                ts=collected_at,
                lang="en",
                source_type="github",
                source_tier="A",
                metadata={
                    "full_name": repo.full_name,
                    "stars": str(repo.stars),
                    "language": repo.language,
                },
            )
        )
    return items


def _rows_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("data", "rows", "items", "repos"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("rows", "items", "repos"):
            value = data.get(key)
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
