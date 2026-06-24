"""People/alias evidence contract for dialogue memory."""

from __future__ import annotations

from pathlib import Path


def test_extract_explicit_username_alias_candidate() -> None:
    from src.dialogue.people import extract_people_alias_candidates

    candidates = extract_people_alias_candidates(
        "@Anroxa2748 это Тоша",
        source_message_id="tg:10",
    )

    assert len(candidates) == 1
    assert candidates[0].alias == "Тоша"
    assert candidates[0].executable_chat_id == "@Anroxa2748"
    assert candidates[0].source_message_id == "tg:10"
    assert candidates[0].confidence >= 0.8


def test_alias_lookup_requires_confirmation_and_never_executes(tmp_path: Path) -> None:
    from src.dialogue.people import (
        FilePeopleAliasStore,
        extract_people_alias_candidates,
    )

    store = FilePeopleAliasStore(tmp_path)
    candidate = extract_people_alias_candidates(
        "@Anroxa2748 это Тоша",
        source_message_id="tg:10",
    )[0]
    store.append(chat_id=12345, candidate=candidate)

    reloaded = FilePeopleAliasStore(tmp_path)
    result = reloaded.lookup(chat_id=12345, alias="Тоше")

    assert result.status == "needs_confirmation"
    assert result.alias == "Тоше"
    assert result.suggested_recipient == "@Anroxa2748"
    assert result.can_execute is False
    assert result.missing_fields == ("chat_id",)
    assert result.candidates[0].alias == "Тоша"


def test_alias_extractor_ignores_human_hint_without_explicit_username() -> None:
    from src.dialogue.people import extract_people_alias_candidates

    assert extract_people_alias_candidates("пиши Тоше") == ()


def test_alias_lookup_rejects_low_confidence_candidate(tmp_path: Path) -> None:
    from src.dialogue.people import FilePeopleAliasStore, PeopleAliasCandidate

    store = FilePeopleAliasStore(tmp_path)
    store.append(
        chat_id=12345,
        candidate=PeopleAliasCandidate(
            alias="Тоша",
            executable_chat_id="@Anroxa2748",
            source_text="@Anroxa2748 возможно Тоша",
            confidence=0.4,
        ),
    )

    result = store.lookup(chat_id=12345, alias="Тоше", min_confidence=0.7)

    assert result.status == "insufficient_confidence"
    assert result.suggested_recipient == ""
    assert result.can_execute is False
    assert result.candidates[0].executable_chat_id == "@Anroxa2748"


def test_alias_lookup_rejects_stale_candidate(tmp_path: Path) -> None:
    from src.dialogue.people import FilePeopleAliasStore, PeopleAliasCandidate

    store = FilePeopleAliasStore(tmp_path)
    store.append(
        chat_id=12345,
        candidate=PeopleAliasCandidate(
            alias="Тоша",
            executable_chat_id="@Anroxa2748",
            source_text="@Anroxa2748 это Тоша",
            confidence=0.95,
            observed_at="2000-01-01T00:00:00+00:00",
        ),
    )

    result = store.lookup(chat_id=12345, alias="Тоше", max_age_days=30)

    assert result.status == "stale"
    assert result.suggested_recipient == ""
    assert result.can_execute is False
    assert result.candidates[0].executable_chat_id == "@Anroxa2748"


def test_render_people_alias_lookup_status_hides_raw_source_text(
    tmp_path: Path,
) -> None:
    from src.dialogue.people import (
        FilePeopleAliasStore,
        PeopleAliasCandidate,
        render_people_alias_lookup_status,
    )

    store = FilePeopleAliasStore(tmp_path)
    store.append(
        chat_id=12345,
        candidate=PeopleAliasCandidate(
            alias="Тоша",
            executable_chat_id="@Anroxa2748",
            source_text="@Anroxa2748 это Тоша и секретный контекст",
            confidence=0.95,
        ),
    )

    status = render_people_alias_lookup_status(
        store.lookup(chat_id=12345, alias="Тоше")
    )

    assert "People alias lookup:" in status
    assert "status: needs_confirmation" in status
    assert "suggested_recipient: @Anroxa2748" in status
    assert "confidence=0.95" in status
    assert "секретный контекст" not in status
