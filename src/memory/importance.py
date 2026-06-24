"""Rule-based importance scoring for episodes. No LLM calls."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from src.embeddings import EmbeddingService

if TYPE_CHECKING:
    from src.memory.protocols import Episode


# Evaluative language patterns (Russian)
_EVALUATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bхорош[оа]?\b",
        r"\bплох[оа]?\b",
        r"\bне надо\b",
        r"\bкруто\b",
        r"\bотлично\b",
        r"\bужасно\b",
        r"\bсупер\b",
        r"\bнорм\b",
        r"\bок(ей)?\b",
        r"\bтак\s+не\s+надо\b",
        r"\bправильно\b",
        r"\bнеправильно\b",
    ]
]

# "Remember this" patterns
_REMEMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bзапомни\b",
        r"\bпомни\b",
        r"\bremember\b",
    ]
]

# Profanity / aggressive markers (Russian)
_PROFANITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bблять?\b",
        r"\bсука\b",
        r"\bпиздец\b",
        r"\bхуй\b",
        r"\bебать?\b",
        r"\bнахуй\b",
    ]
]


class ImportanceScorer:
    """Scores importance of new episodes.

    Implements predictive coding principle: surprise > routine.
    Rule-based — no LLM calls.
    """

    ADMIN_FEEDBACK_BOOST: float = 0.3
    NOVELTY_BOOST: float = 0.2
    EXPLICIT_REMEMBER_SCORE: float = 1.0

    _BASE_SCORES: ClassVar[dict[str, float]] = {
        "personal": 0.5,
        "assistant": 0.3,
        "social": 0.1,
    }

    async def score(
        self,
        content: str,
        user_id: int,
        is_admin: bool,
        chat_type: str,
        recent_episodes: list[Episode],
    ) -> float:
        """Score importance 0.0 to 1.0.

        Base scores by chat_type, modified by admin feedback,
        novelty, explicit "remember" commands, and emotional context.
        """
        # Explicit "remember" overrides everything
        if any(p.search(content) for p in _REMEMBER_PATTERNS):
            return self.EXPLICIT_REMEMBER_SCORE

        base = self._BASE_SCORES.get(chat_type, 0.5)

        # Admin evaluative feedback boost
        if is_admin and any(p.search(content) for p in _EVALUATIVE_PATTERNS):
            base += self.ADMIN_FEEDBACK_BOOST

        # Novelty boost via surprise score
        if recent_episodes:
            recent_embeddings = [
                ep.embedding for ep in recent_episodes if ep.embedding is not None
            ]
            if recent_embeddings:
                content_embedding = await EmbeddingService.embed_async(content)
                surprise = self._surprise_score(content_embedding, recent_embeddings)
                if surprise > 0.7:
                    base += self.NOVELTY_BOOST

        # Emotional context penalty
        base += self._detect_emotional_context(content)

        return max(0.0, min(1.0, base))

    def _surprise_score(
        self,
        embedding: list[float],
        recent_embeddings: list[list[float]],
    ) -> float:
        """How surprising is this compared to recent context.

        Returns 0.0 (completely expected) to 1.0 (totally new).
        """
        if not recent_embeddings:
            return 1.0

        similarities = [
            EmbeddingService.cosine_similarity(embedding, recent)
            for recent in recent_embeddings[:10]
        ]
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity

    def _detect_emotional_context(self, content: str) -> float:
        """Detect if message was written in anger/frustration.

        Returns penalty 0.0 (calm) to -0.15 (angry).
        """
        penalty = 0.0

        # ALL CAPS words count > 3
        caps_words = [w for w in content.split() if w.isupper() and len(w) > 1]
        if len(caps_words) > 3:
            penalty -= 0.1

        # Multiple exclamation marks
        if re.search(r"[!?]{3,}", content):
            penalty -= 0.05

        # Short aggressive phrases (< 5 words with !)
        sentences = content.split(".")
        for sentence in sentences:
            stripped = sentence.strip()
            if stripped.endswith("!") and len(stripped.split()) < 5:
                penalty -= 0.05
                break

        # Profanity
        if any(p.search(content) for p in _PROFANITY_PATTERNS):
            penalty -= 0.1

        return max(-0.15, penalty)
