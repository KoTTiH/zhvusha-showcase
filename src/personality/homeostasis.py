"""Homeostasis check — prevents personality from drifting to extremes.

Runs during morning consolidation after Phase 3.
Compares recent behavior against genes.md baseline.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.personality.protocols import HomeostasisCorrection, HomeostasisProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory import Episode

__all__ = ["HomeostasisCheck", "HomeostasisCorrection"]


# Gene baseline expectations
_GENE_CHECKS: dict[str, dict[str, object]] = {
    "initiative": {
        "expected_proposals_per_week": 2,
        "low_evidence": "proposals in recent episodes",
    },
    "honesty": {
        "sycophancy_patterns": [
            r"(?:конечно|безусловно|ты.*прав[аы]?|ты.*молодец)[\s!]",
        ],
    },
    "caution": {
        "expected_approval_rate": 0.7,
    },
    "energy": {
        "min_avg_response_length": 50,
    },
    "emotional_stability": {
        "max_negative_streak": 5,
        "max_arousal_spikes": 3,
    },
}


class HomeostasisCheck(HomeostasisProtocol):
    """Prevents personality from drifting to extremes."""

    async def check(
        self,
        genes_path: Path,
        recent_episodes: list[Episode],
        admin_user_id: int,
    ) -> list[HomeostasisCorrection]:
        """Check for extreme drift against genes.md baseline."""
        if not recent_episodes:
            return []

        corrections: list[HomeostasisCorrection] = []

        # Initiative check: count proactive proposals
        proposals = self._count_proposals(recent_episodes, admin_user_id)
        if proposals == 0 and len(recent_episodes) > 20:
            corrections.append(
                HomeostasisCorrection(
                    gene="initiative",
                    direction="too_low",
                    evidence=f"0 предложений в {len(recent_episodes)} эпизодах",
                    suggestion="Попробовать предложить что-нибудь проактивно завтра",
                )
            )

        # Honesty check: sycophantic patterns
        syc_count = self._count_sycophantic(recent_episodes, admin_user_id)
        total_responses = sum(1 for ep in recent_episodes if ep.role == "assistant")
        if total_responses > 5 and syc_count / total_responses > 0.5:
            corrections.append(
                HomeostasisCorrection(
                    gene="honesty",
                    direction="too_high",
                    evidence=(
                        f"Сикофантные паттерны в {syc_count}/{total_responses} ответах"
                    ),
                    suggestion="Быть более прямой и честной, меньше соглашаться",
                )
            )

        # Caution check: approval rate
        approved, rejected = self._count_approvals(recent_episodes)
        total_decisions = approved + rejected
        if total_decisions >= 5:
            rate = approved / total_decisions
            if rate < 0.5:
                corrections.append(
                    HomeostasisCorrection(
                        gene="caution",
                        direction="too_low",
                        evidence=f"Процент одобрений {rate:.0%} ({approved}/{total_decisions})",
                        suggestion="Быть аккуратнее с предложениями, сначала спрашивать Никиту",
                    )
                )

        # Emotional stability check
        corrections.extend(
            self._check_emotional_stability(recent_episodes, admin_user_id)
        )

        # Energy check: average response length
        response_lengths = [
            len(ep.content) for ep in recent_episodes if ep.role == "assistant"
        ]
        if response_lengths:
            avg_len = sum(response_lengths) / len(response_lengths)
            if avg_len < 30:
                corrections.append(
                    HomeostasisCorrection(
                        gene="energy",
                        direction="too_low",
                        evidence=f"Средняя длина ответа: {avg_len:.0f} символов",
                        suggestion="Включаться активнее, давать полные ответы",
                    )
                )

        return corrections

    @staticmethod
    def _count_proposals(episodes: list[Episode], admin_user_id: int) -> int:
        """Count proactive proposals (assistant messages with proposing language)."""
        proposal_words = (
            "предлагаю",
            "давай",
            "может",
            "стоит",
            "попробуем",
            "а если",
        )
        count = 0
        for ep in episodes:
            if ep.role == "assistant" and ep.user_id == admin_user_id:
                lower = ep.content.lower()
                if any(w in lower for w in proposal_words):
                    count += 1
        return count

    @staticmethod
    def _count_sycophantic(episodes: list[Episode], admin_user_id: int) -> int:
        """Count responses with sycophantic patterns."""
        sycophancy_list = _GENE_CHECKS["honesty"]["sycophancy_patterns"]
        assert isinstance(sycophancy_list, list)  # type guard
        patterns = [re.compile(str(p), re.IGNORECASE) for p in sycophancy_list]
        count = 0
        for ep in episodes:
            if (
                ep.role == "assistant"
                and ep.user_id == admin_user_id
                and any(p.search(ep.content) for p in patterns)
            ):
                count += 1
        return count

    @staticmethod
    def _check_emotional_stability(
        episodes: list[Episode],
        admin_user_id: int,
    ) -> list[HomeostasisCorrection]:
        """Check for emotional drift: negative streaks or arousal spikes."""
        corrections: list[HomeostasisCorrection] = []
        assistant_eps = [
            ep
            for ep in episodes
            if ep.role == "assistant" and ep.user_id == admin_user_id
        ]
        if not assistant_eps:
            return corrections

        # Check 1: negative streak (consecutive negative-valence responses)
        consecutive_negative = 0
        for ep in reversed(assistant_eps):
            if ep.valence == "negative":
                consecutive_negative += 1
            else:
                break
        if consecutive_negative >= 5:
            corrections.append(
                HomeostasisCorrection(
                    gene="emotional_stability",
                    direction="too_low",
                    evidence=(f"{consecutive_negative} подряд негативных ответов"),
                    suggestion=(
                        "Сбросить эмоциональное состояние к baseline, "
                        "начать с чего-то позитивного"
                    ),
                )
            )

        # Check 2: arousal spikes from enrichment metadata
        high_arousal_count = 0
        for ep in assistant_eps:
            meta_str = getattr(ep, "metadata_json", None)
            if not meta_str:
                continue
            try:
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                enrichment = meta.get("enrichment", {})
                self_arousal = enrichment.get("self_arousal", 0.5)
                if self_arousal > 0.8:
                    high_arousal_count += 1
            except (json.JSONDecodeError, AttributeError):
                continue

        if high_arousal_count > 3:
            corrections.append(
                HomeostasisCorrection(
                    gene="emotional_stability",
                    direction="too_high",
                    evidence=(f"{high_arousal_count} эпизодов с высоким arousal"),
                    suggestion=("Снизить интенсивность реакций, быть спокойнее"),
                )
            )

        return corrections

    @staticmethod
    def _count_approvals(
        episodes: list[Episode],
    ) -> tuple[int, int]:
        """Count approved vs rejected actions."""
        approved = 0
        rejected = 0
        for ep in episodes:
            if ep.valence == "positive" and ep.role == "user":
                approved += 1
            elif ep.valence == "negative" and ep.role == "user":
                rejected += 1
        return approved, rejected
