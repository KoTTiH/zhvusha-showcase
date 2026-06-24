"""Memory database layer — public re-exports for legitimate DB clients.

Public surface:
    * ``Base`` — shared SQLAlchemy declarative base. Used by ORM model
      definitions in :mod:`src.daemon.audit`, :mod:`src.daemon.pending_action`,
      :mod:`src.knowledge.models`, and by :mod:`alembic.env` for metadata
      discovery. Legitimate shared SQLAlchemy infrastructure.
    * ``EpisodeORM`` — ORM model (renamed from ``Episode`` in phase 5D).
    * ``Episode`` — backward-compat alias for ``EpisodeORM`` preserved
      for existing callers (alembic, :mod:`src.mcp_server.dashboard_api`,
      tests, scripts). New code should prefer ``EpisodeORM`` when
      interacting with the ORM, and ``src.memory.protocols.Episode``
      (frozen dataclass) when only the domain object is needed.
    * ``get_engine`` / ``get_session_maker`` — engine/session factories
      for legitimate direct DB access clients (:mod:`src.mcp_server.server`,
      :mod:`src.daemon.main`, :mod:`src.bot.main`, migration scripts).

Internal modules ``_models`` and ``_engine`` are **forbidden** to outside
clients by the ``memory_isolation`` rule in ``.importlinter``. Only this
package ``__init__`` re-exports from them.
"""

from __future__ import annotations

from src.memory.database._engine import get_engine, get_session_maker
from src.memory.database._models import Base, EpisodeORM

Episode = EpisodeORM

__all__ = [
    "Base",
    "Episode",
    "EpisodeORM",
    "get_engine",
    "get_session_maker",
]
