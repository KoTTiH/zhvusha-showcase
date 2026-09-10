"""Embeddings leaf module.

Public API: :class:`EmbeddingService` for sentence-transformer based
vector embeddings. This is a **leaf module** — it does not import from
any other ``src/`` module. Only depends on ``sentence-transformers``
(lazy-loaded) and ``numpy`` (lazy-loaded).

Clients import from the package, not from the submodule::

    from src.embeddings import EmbeddingService

    # NOT:
    from src.embeddings.service import EmbeddingService

See also: KB records #69 (v4 architecture), #82 (enforcement),
#97 (lessons learned).
"""

from src.embeddings.service import EmbeddingService

__all__ = ["EmbeddingService"]
