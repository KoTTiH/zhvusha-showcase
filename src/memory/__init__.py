"""Memory capability module (v4).

Public contract: :mod:`src.memory.protocols`.
Concrete implementations: :mod:`src.memory.episodic`,
:mod:`src.memory.people`, :mod:`src.memory.desires`,
:mod:`src.memory.domain`, :mod:`src.memory.consolidation`,
:mod:`src.memory.sonnet_enricher`, :mod:`src.memory.learning_staging`,
:mod:`src.memory.consolidation_lock`.

Other modules MUST import from this package, not from internal submodules.
The ``memory_isolation`` rule in ``.importlinter`` enforces this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.consolidation import ConsolidationEngine
from src.memory.consolidation_lock import ConsolidationLock
from src.memory.desires import DesireProcessor
from src.memory.episodic import EpisodicMemory
from src.memory.learning_staging import StagingWriter
from src.memory.people import PeopleManager, get_people_manager
from src.memory.protocols import (
    ChatType,
    ConsolidationAction,
    ConsolidationProtocol,
    ConsolidationResult,
    DesireProcessorProtocol,
    Domain,
    EnrichmentProtocol,
    EnrichmentResult,
    Episode,
    EpisodeNotFoundError,
    EpisodicMemoryProtocol,
    LearningSignal,
    MemoryModuleError,
    PeopleManagerProtocol,
    PersonNotFoundError,
    PersonProfile,
    Role,
    StagingWriterProtocol,
    Valence,
    detect_domain,
)
from src.memory.sonnet_enricher import SonnetEnricher, get_enricher
from src.memory.types import parse_enrichment_json

if TYPE_CHECKING:
    from pathlib import Path


def get_staging_writer(staging_dir: Path) -> StagingWriterProtocol:
    """Default factory for :class:`StagingWriterProtocol` implementations.

    Returns a fresh :class:`StagingWriter` bound to ``staging_dir``.
    Clients that need a custom writer can inject their own
    :class:`StagingWriterProtocol`; otherwise call this factory.
    """
    return StagingWriter(staging_dir)


__all__ = [
    "ChatType",
    "ConsolidationAction",
    "ConsolidationEngine",
    "ConsolidationLock",
    "ConsolidationProtocol",
    "ConsolidationResult",
    "DesireProcessor",
    "DesireProcessorProtocol",
    "Domain",
    "EnrichmentProtocol",
    "EnrichmentResult",
    "Episode",
    "EpisodeNotFoundError",
    "EpisodicMemory",
    "EpisodicMemoryProtocol",
    "LearningSignal",
    "MemoryModuleError",
    "PeopleManager",
    "PeopleManagerProtocol",
    "PersonNotFoundError",
    "PersonProfile",
    "Role",
    "SonnetEnricher",
    "StagingWriter",
    "StagingWriterProtocol",
    "Valence",
    "detect_domain",
    "get_enricher",
    "get_people_manager",
    "get_staging_writer",
    "parse_enrichment_json",
]
