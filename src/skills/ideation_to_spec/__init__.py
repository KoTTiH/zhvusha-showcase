"""ideation_to_spec — Architect skill (Phase 12).

Currently exports only the deterministic ``classify_spec_tier`` algorithm. The
DelegatedSkill class itself is implemented in :mod:`.skill` (in flight).
"""

from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

__all__ = ["classify_spec_tier"]
