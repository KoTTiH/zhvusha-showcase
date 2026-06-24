"""Contract tests for the Memory capability module (phase 5b).

Verifies that concrete implementations conform to their protocols
(``EpisodicMemory`` → ``EpisodicMemoryProtocol``, ``PeopleManager`` →
``PeopleManagerProtocol``, ``DesireProcessor`` → ``DesireProcessorProtocol``)
and that the public domain types have the expected shape. Uses ``MagicMock``
for external dependencies — **no** real DB / embeddings calls happen here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.memory import (
    DesireProcessor,
    DesireProcessorProtocol,
    Episode,
    EpisodeNotFoundError,
    EpisodicMemory,
    EpisodicMemoryProtocol,
    MemoryModuleError,
    PeopleManager,
    PeopleManagerProtocol,
    PersonNotFoundError,
    detect_domain,
)
from src.memory.database import Episode as EpisodeORM

pytestmark = pytest.mark.contract


# === Helpers ===


def _make_episodic() -> EpisodicMemory:
    """Build an EpisodicMemory with a mocked session maker."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    maker: Any = MagicMock(return_value=session)
    return EpisodicMemory(maker, admin_user_id=42)


def _make_people(tmp_path: Path) -> PeopleManager:
    return PeopleManager(workspace_root=tmp_path)


def _make_desire(tmp_path: Path) -> DesireProcessor:
    return DesireProcessor(workspace_root=tmp_path, episodic=None)


# === Protocol conformance ===


class TestProtocolConformance:
    """Concrete implementations satisfy their runtime_checkable protocol."""

    def test_episodic_memory_is_protocol_instance(self) -> None:
        assert isinstance(_make_episodic(), EpisodicMemoryProtocol)

    def test_people_manager_is_protocol_instance(self, tmp_path: Path) -> None:
        assert isinstance(_make_people(tmp_path), PeopleManagerProtocol)

    def test_desire_processor_is_protocol_instance(self, tmp_path: Path) -> None:
        assert isinstance(_make_desire(tmp_path), DesireProcessorProtocol)


# === EpisodicMemoryProtocol surface ===


class TestEpisodicMemoryContract:
    """Ten async methods with documented signatures."""

    def test_public_methods_present(self) -> None:
        expected = {
            "record",
            "retrieve",
            "retrieve_by_somatic_marker",
            "complete_pattern",
            "check_pattern_separation",
            "get_unconsolidated",
            "mark_consolidated",
            "update_importance",
            "update_valence",
            "update_enrichment",
        }
        em = _make_episodic()
        for name in expected:
            method = getattr(em, name, None)
            assert method is not None, f"Missing protocol method: {name}"
            assert callable(method), f"Protocol method not callable: {name}"

    def test_exactly_ten_protocol_methods(self) -> None:
        """EpisodicMemoryProtocol exposes exactly 10 public methods."""
        public = [
            m
            for m in dir(EpisodicMemoryProtocol)
            if not m.startswith("_") and callable(getattr(EpisodicMemoryProtocol, m))
        ]
        assert len(public) == 10, f"Expected 10 methods, got {len(public)}: {public}"

    async def test_record_is_callable_and_async(self) -> None:
        """``record`` is an async callable accepting the documented params.

        We don't assert on the return value here — the happy-path int id
        comes from a real DB flush and cannot be synthesised via MagicMock
        without over-mocking. ``test_record_social_rate_limit_returns_minus_one``
        exercises the one deterministic branch.
        """
        import inspect

        assert inspect.iscoroutinefunction(EpisodicMemory.record)

    async def test_record_social_rate_limit_returns_minus_one(self) -> None:
        em = _make_episodic()
        for _ in range(11):  # _SOCIAL_RATE_LIMIT is 10
            await em.record(content="x", user_id=7, chat_type="social", role="user")
        result = await em.record(
            content="x", user_id=7, chat_type="social", role="user"
        )
        assert result == -1

    async def test_retrieve_returns_list(self) -> None:
        em = _make_episodic()
        maker: Any = em.session_maker
        maker.return_value.__aenter__.return_value.execute.return_value = MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=lambda: []))
        )
        result = await em.retrieve("anything")
        assert isinstance(result, list)

    async def test_mark_consolidated_empty_noop(self) -> None:
        em = _make_episodic()
        # Should not raise on empty list (early return in implementation).
        result = await em.mark_consolidated([])
        assert result is None


# === PeopleManagerProtocol surface ===


class TestPeopleManagerContract:
    """Six synchronous methods."""

    def test_public_methods_present(self, tmp_path: Path) -> None:
        expected = {
            "get_or_create_profile",
            "update_profile",
            "record_interaction",
            "get_interaction_count",
            "get_significance_level",
            "get_profile_for_context",
        }
        pm = _make_people(tmp_path)
        for name in expected:
            method = getattr(pm, name, None)
            assert method is not None, f"Missing protocol method: {name}"
            assert callable(method), f"Protocol method not callable: {name}"

    def test_get_or_create_profile_returns_dict(self, tmp_path: Path) -> None:
        pm = _make_people(tmp_path)
        profile = pm.get_or_create_profile(user_id=1, username="nikita")
        assert isinstance(profile, dict)
        # PersonProfile TypedDict keys
        for key in (
            "user_id",
            "username",
            "first_name",
            "significance",
            "interaction_count",
            "first_seen",
            "last_seen",
        ):
            assert key in profile, f"Missing PersonProfile key: {key}"

    def test_record_interaction_returns_bool(self, tmp_path: Path) -> None:
        pm = _make_people(tmp_path)
        pm.get_or_create_profile(user_id=1)
        result = pm.record_interaction(user_id=1)
        assert isinstance(result, bool)

    def test_get_interaction_count_returns_int(self, tmp_path: Path) -> None:
        pm = _make_people(tmp_path)
        assert pm.get_interaction_count(user_id=999) == 0

    def test_get_significance_level_default_stranger(self, tmp_path: Path) -> None:
        pm = _make_people(tmp_path)
        assert pm.get_significance_level(user_id=999) == "stranger"

    def test_get_profile_for_context_social_returns_empty(self, tmp_path: Path) -> None:
        pm = _make_people(tmp_path)
        pm.get_or_create_profile(user_id=1)
        result = pm.get_profile_for_context(user_id=1, mode="social")
        assert result == ""

    def test_module_has_no_skills_dependency(self) -> None:
        """``src.memory.people`` must not import from ``src.skills``.

        Memory sits below Skills in the layered architecture, so any import
        in this direction is a ``layered_architecture`` violation. Historical
        coupling (``get_workspace_path`` reach-in) has been grandfathered in
        ``.importlinter`` — this test guards the cleanup.
        """
        import inspect

        from src.memory import people as people_module

        source = inspect.getsource(people_module)
        assert "from src.skills" not in source, (
            "memory.people imports from src.skills — layered_architecture violation"
        )
        assert "import src.skills" not in source


# === DesireProcessorProtocol surface ===


class TestDesireProcessorContract:
    """Single async method."""

    def test_public_methods_present(self, tmp_path: Path) -> None:
        dp = _make_desire(tmp_path)
        assert callable(getattr(dp, "run_all", None))

    async def test_run_all_returns_str(self, tmp_path: Path) -> None:
        dp = _make_desire(tmp_path)
        result = await dp.run_all()
        assert isinstance(result, str)


# === Episode domain type ===


class TestEpisodeDomainType:
    """Episode is a frozen dataclass mirroring the ORM columns."""

    def test_episode_is_frozen(self) -> None:
        ep = Episode(
            id=1,
            timestamp=__import__("datetime").datetime.now(),
            user_id=7,
            chat_type="personal",
            role="user",
            content="hi",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ep.id = 999  # type: ignore[misc]

    def test_episode_field_parity_with_orm(self) -> None:
        """Domain Episode mirrors ORM Episode columns 1:1."""
        orm_columns = {c.name for c in EpisodeORM.__table__.columns}
        domain_fields = {f.name for f in dataclasses.fields(Episode)}
        assert orm_columns == domain_fields, (
            f"Field parity broken. Missing in domain: {orm_columns - domain_fields}. "
            f"Extra in domain: {domain_fields - orm_columns}"
        )

    def test_episode_has_21_fields(self) -> None:
        assert len(dataclasses.fields(Episode)) == 21

    def test_episode_defaults(self) -> None:
        ep = Episode(
            id=1,
            timestamp=__import__("datetime").datetime.now(),
            user_id=7,
            chat_type="personal",
            role="user",
            content="hi",
        )
        assert ep.importance == 0.5
        assert ep.valence == "neutral"
        assert ep.consolidated is False
        assert ep.access_count == 0
        assert ep.enrichment_status == "pending"


# === Error hierarchy ===


class TestErrorHierarchy:
    def test_episode_not_found_is_memory_error(self) -> None:
        assert issubclass(EpisodeNotFoundError, MemoryModuleError)

    def test_person_not_found_is_memory_error(self) -> None:
        assert issubclass(PersonNotFoundError, MemoryModuleError)

    def test_memory_module_error_is_exception(self) -> None:
        assert issubclass(MemoryModuleError, Exception)


# === detect_domain ===


class TestDetectDomain:
    def test_kwork_via_keyword(self) -> None:
        assert detect_domain("мне заказ на Kwork пришёл") == "kwork"

    def test_source_overrides(self) -> None:
        assert detect_domain("anything", source="kwork") == "kwork"
        assert detect_domain("anything", source="channel") == "content"

    def test_assistant_mode_is_outreach(self) -> None:
        assert detect_domain("кто-то пишет", source="", mode="assistant") == "outreach"

    def test_default_is_chat(self) -> None:
        assert detect_domain("просто разговор") == "chat"


# === Protocol method signatures have type hints ===


class TestProtocolSignatures:
    """Sanity: protocol methods carry type hints so mypy strict can verify."""

    def test_episodic_record_signature_hints(self) -> None:
        hints = get_type_hints(EpisodicMemoryProtocol.record)
        assert "content" in hints
        assert "user_id" in hints
        assert hints["return"] is int

    def test_people_get_or_create_signature_hints(self) -> None:
        hints = get_type_hints(PeopleManagerProtocol.get_or_create_profile)
        assert "user_id" in hints
