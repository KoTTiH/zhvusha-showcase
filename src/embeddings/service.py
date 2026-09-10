"""Embedding service using sentence-transformers (lazy-loaded)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class EmbeddingService:
    """Generates 384-dim embeddings using sentence-transformers.

    Loads model lazily on first call (~2sec). Model stays in memory
    for subsequent calls (~5ms per embed). Uses paraphrase-multilingual-MiniLM-L12-v2
    (~471MB, runs on CPU, no GPU needed). Multilingual model for Russian content.
    """

    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    from sentence_transformers import SentenceTransformer

                    cls._model = SentenceTransformer(
                        "paraphrase-multilingual-MiniLM-L12-v2"
                    )
        return cls._model

    @classmethod
    def embed(cls, text: str) -> list[float]:
        """Generate 384-dim embedding for text."""
        model = cls.get_model()
        return model.encode(text).tolist()  # type: ignore[no-any-return]

    @classmethod
    async def embed_async(cls, text: str) -> list[float]:
        """Non-blocking embed — runs model inference in thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.embed, text)

    @classmethod
    def embed_batch(cls, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        model = cls.get_model()
        return [v.tolist() for v in model.encode(texts)]

    @classmethod
    def cosine_similarity(cls, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np

        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        dot = float(np.dot(a_np, b_np))
        norm_a = float(np.linalg.norm(a_np))
        norm_b = float(np.linalg.norm(b_np))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
