"""Tests for DesireProcessor (morning analytics)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.memory.desires import (
    DesireProcessor,
    _parse_dreams,
    _parse_wishlist_sections,
)

# --- Parser tests ---


def test_parse_dreams_valid() -> None:
    text = """\
# Мечты Жвуши

- [2026-03-20] научиться рисовать
- [2026-03-25] написать свою книгу
"""
    dreams = _parse_dreams(text)
    assert len(dreams) == 2
    assert dreams[0].date == date(2026, 3, 20)
    assert dreams[0].text == "научиться рисовать"
    assert dreams[1].text == "написать свою книгу"


def test_parse_dreams_empty() -> None:
    assert _parse_dreams("# Мечты Жвуши\n\n_Пока нет._\n") == []


def test_parse_dreams_invalid_date() -> None:
    text = "- [invalid-date] something\n- [2026-03-20] valid\n"
    dreams = _parse_dreams(text)
    assert len(dreams) == 1
    assert dreams[0].text == "valid"


def test_parse_wishlist_sections() -> None:
    text = """\
# Wishlist

## Хочу

- Научиться рисовать
- Написать книгу

## В работе

- Сделать бота

## Готово

- Запустить канал
"""
    sections = _parse_wishlist_sections(text)
    assert len(sections["Хочу"]) == 2
    assert len(sections["В работе"]) == 1
    assert len(sections["Готово"]) == 1


# --- DesireProcessor tests ---


@pytest.fixture
def workspace(tmp_path):  # type: ignore[no-untyped-def]
    personality = tmp_path / "personality"
    personality.mkdir()
    (personality / "dreams.md").write_text("# Мечты\n\n", encoding="utf-8")
    (personality / "wishlist.md").write_text(
        "# Wishlist\n\n## Хочу\n\n## В работе\n\n## Готово\n",
        encoding="utf-8",
    )
    (tmp_path / "outbox" / "dream_candidates").mkdir(parents=True)
    (personality / "history").mkdir()
    return tmp_path


@pytest.mark.asyncio
async def test_crystallization_with_episodic(workspace) -> None:  # type: ignore[no-untyped-def]
    """Dream >7 days with 3+ episodic matches becomes candidate."""
    old_date = (datetime.now(tz=UTC).date() - timedelta(days=10)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    dreams_path.write_text(
        f"# Мечты\n\n- [{old_date}] научиться рисовать\n",
        encoding="utf-8",
    )

    episodic = AsyncMock()
    episodic.retrieve = AsyncMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

    proc = DesireProcessor(workspace, episodic=episodic)
    result = await proc._check_crystallization()
    assert "научиться рисовать" in result

    # Candidate file should be created
    candidates = list((workspace / "outbox" / "dream_candidates").iterdir())
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_crystallization_no_episodic(workspace) -> None:  # type: ignore[no-untyped-def]
    """Without episodic, crystallization is skipped."""
    proc = DesireProcessor(workspace, episodic=None)
    result = await proc._check_crystallization()
    assert result == ""


@pytest.mark.asyncio
async def test_crystallization_young_dream(workspace) -> None:  # type: ignore[no-untyped-def]
    """Dreams <7 days are skipped."""
    recent_date = (datetime.now(tz=UTC).date() - timedelta(days=3)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    dreams_path.write_text(
        f"# Мечты\n\n- [{recent_date}] новая мечта\n",
        encoding="utf-8",
    )

    episodic = AsyncMock()
    proc = DesireProcessor(workspace, episodic=episodic)
    result = await proc._check_crystallization()
    assert result == ""
    episodic.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_crystallization_few_matches(workspace) -> None:  # type: ignore[no-untyped-def]
    """Dreams with <3 episodic matches are skipped."""
    old_date = (datetime.now(tz=UTC).date() - timedelta(days=10)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    dreams_path.write_text(
        f"# Мечты\n\n- [{old_date}] мечта\n",
        encoding="utf-8",
    )

    episodic = AsyncMock()
    episodic.retrieve = AsyncMock(return_value=[MagicMock()])  # only 1

    proc = DesireProcessor(workspace, episodic=episodic)
    result = await proc._check_crystallization()
    assert result == ""


def test_escalate_stale(workspace) -> None:  # type: ignore[no-untyped-def]
    """Dreams >15 days are escalated."""
    old_date = (datetime.now(tz=UTC).date() - timedelta(days=20)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    dreams_path.write_text(
        f"# Мечты\n\n- [{old_date}] застоявшаяся мечта\n",
        encoding="utf-8",
    )

    proc = DesireProcessor(workspace)
    result = proc._escalate_stale()
    assert "застоявшаяся мечта" in result
    assert "⏰" in result


def test_escalate_stale_no_stale(workspace) -> None:  # type: ignore[no-untyped-def]
    """No stale dreams returns empty string."""
    recent_date = (datetime.now(tz=UTC).date() - timedelta(days=5)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    dreams_path.write_text(
        f"# Мечты\n\n- [{recent_date}] свежая мечта\n",
        encoding="utf-8",
    )

    proc = DesireProcessor(workspace)
    result = proc._escalate_stale()
    assert result == ""


def test_condense_dreams_below_threshold(workspace) -> None:  # type: ignore[no-untyped-def]
    """<5 dreams returns empty string."""
    proc = DesireProcessor(workspace)
    result = proc._condense_dreams()
    assert result == ""


def test_condense_dreams_above_threshold(workspace) -> None:  # type: ignore[no-untyped-def]
    """5+ dreams triggers recommendation."""
    dreams_path = workspace / "personality" / "dreams.md"
    lines = ["# Мечты", ""]
    for i in range(6):
        d = (datetime.now(tz=UTC).date() - timedelta(days=i)).isoformat()
        lines.append(f"- [{d}] мечта {i}")
    dreams_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = DesireProcessor(workspace)
    result = proc._condense_dreams()
    assert "6 мечт" in result


def test_enforce_wishlist_within_limits(workspace) -> None:  # type: ignore[no-untyped-def]
    """Wishlist within limits returns empty string."""
    proc = DesireProcessor(workspace)
    result = proc._enforce_wishlist_limits()
    assert result == ""


def test_enforce_wishlist_overflow(workspace) -> None:  # type: ignore[no-untyped-def]
    """Overflow items are archived."""
    wishlist_path = workspace / "personality" / "wishlist.md"
    lines = ["# Wishlist", "", "## Готово", ""]
    for i in range(15):
        lines.append(f"- Задача {i}")
    lines.extend(["", "## Хочу", "", "## В работе", ""])
    wishlist_path.write_text("\n".join(lines), encoding="utf-8")

    proc = DesireProcessor(workspace)
    result = proc._enforce_wishlist_limits()
    assert "архив" in result

    archive = workspace / "personality" / "history" / "wishlist_archive.md"
    assert archive.exists()
    archive_text = archive.read_text(encoding="utf-8")
    assert "Задача" in archive_text


@pytest.mark.asyncio
async def test_run_all_empty_files(workspace) -> None:  # type: ignore[no-untyped-def]
    """run_all on empty files doesn't crash."""
    proc = DesireProcessor(workspace)
    result = await proc.run_all()
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_all_full_cycle(workspace) -> None:  # type: ignore[no-untyped-def]
    """Full run_all with stale dreams produces output."""
    old_date = (datetime.now(tz=UTC).date() - timedelta(days=20)).isoformat()
    dreams_path = workspace / "personality" / "dreams.md"
    lines = ["# Мечты", ""]
    for i in range(6):
        lines.append(f"- [{old_date}] мечта {i}")
    dreams_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = DesireProcessor(workspace)
    result = await proc.run_all()
    assert "⏰" in result  # stale escalation
    assert "мечт" in result  # condensation
