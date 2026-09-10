from __future__ import annotations

from typing import Any, Protocol

from src.skills.kwork_monitor.models import ProjectCard


class KworkProject(Protocol):
    """Duck type for kwork library's WantWorker."""

    id: int | None
    title: str | None
    description: str | None
    price: int | None
    offers: int | None
    username: str | None


def filter_projects(
    projects: list[Any],
    *,
    keywords: list[str],
    min_budget: int,
    max_offers: int,
    seen_ids: set[int],
) -> list[ProjectCard]:
    """Filter raw kwork projects and return matching ProjectCards."""
    result: list[ProjectCard] = []
    keywords_lower = [kw.lower() for kw in keywords]

    for proj in projects:
        project_id: int = proj.id

        if project_id in seen_ids:
            continue

        price: int | None = getattr(proj, "price", None)
        if price is not None and price < min_budget:
            continue

        offers: int | None = getattr(proj, "offers", None)
        if offers is not None and offers > max_offers:
            continue

        title: str = getattr(proj, "title", "") or ""
        description: str = getattr(proj, "description", "") or ""
        searchable = f"{title} {description}".lower()

        matched = [kw for kw in keywords_lower if kw in searchable]
        if not matched:
            continue

        card = ProjectCard(
            id=project_id,
            title=title,
            description=description,
            price=price,
            offers=offers,
            username=getattr(proj, "username", "") or "",
            url=f"https://kwork.ru/projects/{project_id}",
            matched_keywords=matched,
        )
        result.append(card)

    return result
