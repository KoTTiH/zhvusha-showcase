from types import SimpleNamespace

from src.skills.kwork_monitor.filters import filter_projects


def _proj(
    id: int = 1,
    title: str = "Test project",
    description: str = "Need a Python developer",
    price: int | None = 5000,
    offers: int | None = 3,
    username: str = "client1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        title=title,
        description=description,
        price=price,
        offers=offers,
        username=username,
    )


DEFAULTS = {
    "keywords": ["python", "telegram"],
    "min_budget": 3000,
    "max_offers": 15,
    "seen_ids": set(),
}


def test_matches_keyword_in_title():
    projects = [_proj(title="Нужен Python разработчик")]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1
    assert "python" in result[0].matched_keywords


def test_matches_keyword_in_description():
    projects = [_proj(title="Задача", description="Написать Telegram бота")]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1
    assert "telegram" in result[0].matched_keywords


def test_case_insensitive_match():
    projects = [_proj(title="PYTHON бот")]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1


def test_multiple_keywords_matched():
    projects = [_proj(title="Python Telegram бот")]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1
    assert set(result[0].matched_keywords) == {"python", "telegram"}


def test_filters_below_budget():
    projects = [_proj(price=1000)]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 0


def test_filters_above_max_offers():
    projects = [_proj(offers=20)]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 0


def test_skips_seen_ids():
    projects = [_proj(id=42)]
    result = filter_projects(projects, **{**DEFAULTS, "seen_ids": {42}})
    assert len(result) == 0


def test_passes_none_price():
    projects = [_proj(price=None)]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1


def test_passes_none_offers():
    projects = [_proj(offers=None)]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 1


def test_no_keyword_match_filtered():
    projects = [_proj(title="Дизайн логотипа", description="Нужен дизайнер")]
    result = filter_projects(projects, **DEFAULTS)
    assert len(result) == 0


def test_url_format():
    projects = [_proj(id=12345)]
    result = filter_projects(projects, **DEFAULTS)
    assert result[0].url == "https://kwork.ru/projects/12345"
