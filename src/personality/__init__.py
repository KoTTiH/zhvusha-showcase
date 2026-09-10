"""Personality capability module — public API façade.

Public surface (re-exports only):

* **Protocols + domain types** (from :mod:`src.personality.protocols`):
  ``HomeostasisProtocol``, ``PersonalityEvolutionProtocol``,
  ``AffectiveStateProtocol``, ``HomeostasisCorrection``,
  ``AffectiveSnapshot``, ``EmotionConcept``, ``PersonalityError``.
* **Concrete implementations** for DI: ``HomeostasisCheck``,
  ``PersonalityEvolution``, ``AffectiveStateManager``.
* **Singletons / helpers**: ``get_affective_state_manager``,
  ``should_update_personality``.
* **Constants & emotion atlas**: ``PERSONALITY_COMPACT``,
  ``EMOTION_ATLAS``, ``get_complement``, ``get_cluster_members``,
  ``get_decay_target``.

Internal modules (``affective_state``, ``constants``, ``emotion_atlas``,
``evolution``, ``guard``, ``homeostasis``) are **forbidden** to outside
clients by the ``personality_isolation`` rule in ``.importlinter``. Only
this package ``__init__`` re-exports from them.
"""

from __future__ import annotations

from src.personality.affective_state import (
    AffectiveStateManager,
    get_affective_state_manager,
)
from src.personality.constants import PERSONALITY_COMPACT
from src.personality.emotion_atlas import (
    EMOTION_ATLAS,
    get_cluster_members,
    get_complement,
    get_decay_target,
)
from src.personality.evolution import PersonalityEvolution
from src.personality.guard import should_update_personality
from src.personality.homeostasis import HomeostasisCheck
from src.personality.protocols import (
    AffectiveSnapshot,
    AffectiveStateProtocol,
    EmotionConcept,
    HomeostasisCorrection,
    HomeostasisProtocol,
    PersonalityError,
    PersonalityEvolutionProtocol,
)

__all__ = [
    "EMOTION_ATLAS",
    "PERSONALITY_COMPACT",
    "AffectiveSnapshot",
    "AffectiveStateManager",
    "AffectiveStateProtocol",
    "EmotionConcept",
    "HomeostasisCheck",
    "HomeostasisCorrection",
    "HomeostasisProtocol",
    "PersonalityError",
    "PersonalityEvolution",
    "PersonalityEvolutionProtocol",
    "get_affective_state_manager",
    "get_cluster_members",
    "get_complement",
    "get_decay_target",
    "should_update_personality",
]
