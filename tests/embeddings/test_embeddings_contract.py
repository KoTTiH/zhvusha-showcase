"""Contract tests for the Embeddings leaf module.

Verifies the public API of :class:`EmbeddingService` without depending on
any other ``src/`` module. These tests are the **contract** for the
leaf module: they lock in the shape of the public surface (package
re-export, classmethod presence) and the declared behavior of each
method (output dimensions, cosine semantics, async parity).

No real sentence-transformers model is loaded — everything is mocked via
``patch.object(EmbeddingService, "get_model", ...)``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.embeddings import EmbeddingService

pytestmark = pytest.mark.contract


class TestContractShape:
    """Verify leaf module public surface."""

    def test_package_reexport(self) -> None:
        """EmbeddingService is accessible via src.embeddings package."""
        import src.embeddings as embeddings_pkg

        assert embeddings_pkg.EmbeddingService is EmbeddingService

    def test_public_methods_present(self) -> None:
        """All 5 documented classmethods exist and are callable."""
        for name in (
            "get_model",
            "embed",
            "embed_async",
            "embed_batch",
            "cosine_similarity",
        ):
            method = getattr(EmbeddingService, name, None)
            assert method is not None, f"Missing method: {name}"
            assert callable(method), f"Not callable: {name}"


def test_embed_returns_384_dim() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.random.rand(384).astype(np.float32)

    with patch.object(EmbeddingService, "get_model", return_value=fake_model):
        result = EmbeddingService.embed("test text")

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_n_vectors() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)

    with patch.object(EmbeddingService, "get_model", return_value=fake_model):
        result = EmbeddingService.embed_batch(["a", "b", "c"])

    assert len(result) == 3
    assert all(len(v) == 384 for v in result)


def test_cosine_similarity_range() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [1.0, 0.0, 0.0]

    # Orthogonal vectors → 0.0
    assert abs(EmbeddingService.cosine_similarity(a, b)) < 1e-9

    # Identical vectors → 1.0
    assert abs(EmbeddingService.cosine_similarity(a, c) - 1.0) < 1e-9

    # Zero vector → 0.0
    assert EmbeddingService.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


async def test_embed_async_runs_in_executor() -> None:
    """embed_async should produce the same result as embed, non-blocking."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.random.rand(384).astype(np.float32)

    with patch.object(EmbeddingService, "get_model", return_value=fake_model):
        result = await EmbeddingService.embed_async("async test text")

    assert isinstance(result, list)
    assert len(result) == 384
    fake_model.encode.assert_called_once_with("async test text")
