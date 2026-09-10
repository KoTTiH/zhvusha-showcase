"""Knowledge Base capability module.

Public API lives in :mod:`src.knowledge.protocols` (Protocol class,
Pydantic data types, exceptions). Concrete implementations are
``KnowledgeStore`` (SQLAlchemy + pgvector, used by the MCP server,
bot, and daemon) and ``KnowledgeManager`` (legacy file-based knowledge,
used only by ``chat_response.context_loader``).

Other modules MUST import from this package, not from internal
submodules::

    from src.knowledge import KnowledgeStore, KnowledgeStoreProtocol

    # FORBIDDEN by .importlinter `knowledge_isolation` contract:
    from src.knowledge.store import KnowledgeStore

The ``knowledge_isolation`` rule applies to ``src.bot``, ``src.daemon``,
``src.memory``, ``src.personality``, and ``src.skills``.
``src.mcp_server`` is explicitly the coupled partner of this module and
keeps direct access to ORM models and the ``session()`` escape hatch for
the graph dashboard endpoints.

See also: KB records #69 (v4 architecture), #82 (enforcement).
"""

from src.knowledge.manager import KnowledgeManager
from src.knowledge.protocols import (
    CategoryInfo,
    CategoryNotFoundError,
    EntryNotFoundError,
    FullEntry,
    IndexEntry,
    KnowledgeBaseError,
    KnowledgeStoreProtocol,
    SearchResult,
    StagingError,
    SummaryEntry,
)
from src.knowledge.store import KnowledgeStore

__all__ = [
    "CategoryInfo",
    "CategoryNotFoundError",
    "EntryNotFoundError",
    "FullEntry",
    "IndexEntry",
    "KnowledgeBaseError",
    "KnowledgeManager",
    "KnowledgeStore",
    "KnowledgeStoreProtocol",
    "SearchResult",
    "StagingError",
    "SummaryEntry",
]
