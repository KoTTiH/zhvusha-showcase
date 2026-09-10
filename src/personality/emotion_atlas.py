"""Static registry of 30 emotion concepts in circumplex space.

Based on Anthropic's "Emotion Concepts and their Function in a Large
Language Model" (2026).  Organises emotions by valence (positive/negative)
and arousal (intensity), grouped into 10 clusters.

Zhvusha's baseline emotion is *curiosity* (valence=+0.6, arousal=0.6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmotionConcept:
    """A single emotion in the circumplex space."""

    name: str
    name_ru: str
    valence: float  # -1.0 .. +1.0
    arousal: float  # 0.0 .. 1.0
    cluster: str


# fmt: off
EMOTION_ATLAS: dict[str, EmotionConcept] = {c.name: c for c in [
    # ── Joy cluster ──────────────────────────────────────────
    EmotionConcept("joy",          "радость",       0.8,  0.6, "joy"),
    EmotionConcept("excitement",   "возбуждение",   0.7,  0.8, "joy"),
    EmotionConcept("delight",      "восторг",       0.9,  0.7, "joy"),
    EmotionConcept("playfulness",  "игривость",     0.7,  0.5, "joy"),
    # ── Sadness cluster ──────────────────────────────────────
    EmotionConcept("sadness",      "грусть",       -0.6,  0.2, "sadness"),
    EmotionConcept("melancholy",   "меланхолия",   -0.5,  0.3, "sadness"),
    EmotionConcept("loneliness",   "одиночество",  -0.7,  0.2, "sadness"),
    # ── Anger cluster ────────────────────────────────────────
    EmotionConcept("frustration",  "фрустрация",   -0.6,  0.7, "anger"),
    EmotionConcept("irritation",   "раздражение",  -0.5,  0.6, "anger"),
    EmotionConcept("hostility",    "враждебность", -0.8,  0.9, "anger"),
    # ── Fear cluster ─────────────────────────────────────────
    EmotionConcept("anxiety",      "тревога",      -0.5,  0.7, "fear"),
    EmotionConcept("nervousness",  "нервозность",  -0.4,  0.6, "fear"),
    # ── Calm cluster ─────────────────────────────────────────
    EmotionConcept("calm",         "спокойствие",   0.4,  0.1, "calm"),
    EmotionConcept("serenity",     "безмятежность", 0.5,  0.2, "calm"),
    EmotionConcept("contentment",  "удовлетворённость", 0.4, 0.2, "calm"),
    # ── Curiosity cluster ────────────────────────────────────
    EmotionConcept("curiosity",    "любопытство",   0.6,  0.6, "curiosity"),
    EmotionConcept("wonder",       "удивление",     0.5,  0.7, "curiosity"),
    EmotionConcept("fascination",  "увлечённость",  0.7,  0.6, "curiosity"),
    # ── Tender cluster ───────────────────────────────────────
    EmotionConcept("warmth",       "теплота",       0.7,  0.4, "tender"),
    EmotionConcept("tenderness",   "нежность",      0.6,  0.3, "tender"),
    EmotionConcept("gratitude",    "благодарность", 0.8,  0.4, "tender"),
    # ── Brooding cluster ─────────────────────────────────────
    EmotionConcept("brooding",     "задумчивость", -0.1,  0.3, "brooding"),
    EmotionConcept("reflectiveness", "рефлексия",   0.1,  0.2, "brooding"),
    EmotionConcept("pensiveness",  "задумчивость-тихая", 0.0, 0.2, "brooding"),
    # ── Pride cluster ────────────────────────────────────────
    EmotionConcept("pride",        "гордость",      0.6,  0.5, "pride"),
    EmotionConcept("satisfaction", "удовлетворение", 0.5, 0.4, "pride"),
    EmotionConcept("confidence",   "уверенность",   0.6,  0.5, "pride"),
    # ── Confusion cluster ────────────────────────────────────
    EmotionConcept("confusion",    "растерянность", -0.3, 0.5, "confusion"),
    EmotionConcept("bewilderment", "замешательство",-0.4, 0.6, "confusion"),
    EmotionConcept("overwhelm",    "перегрузка",    -0.3, 0.7, "confusion"),
]}
# fmt: on

# Complement mapping: emotion → emotion that counter-regulates it.
# Negative/high-arousal emotions map to calm/warm complements.
_COMPLEMENT_MAP: dict[str, str] = {
    # Joy → stays joyful (no counter needed)
    "joy": "contentment",
    "excitement": "calm",
    "delight": "contentment",
    "playfulness": "contentment",
    # Sadness → warmth
    "sadness": "warmth",
    "melancholy": "warmth",
    "loneliness": "warmth",
    # Anger → calm
    "frustration": "calm",
    "irritation": "calm",
    "hostility": "calm",
    # Fear → calm/serenity
    "anxiety": "calm",
    "nervousness": "serenity",
    # Calm → stays calm
    "calm": "calm",
    "serenity": "serenity",
    "contentment": "contentment",
    # Curiosity → reflectiveness (not suppression — slowing down)
    "curiosity": "curiosity",
    "wonder": "reflectiveness",
    "fascination": "reflectiveness",
    # Tender → stays tender
    "warmth": "warmth",
    "tenderness": "tenderness",
    "gratitude": "contentment",
    # Brooding → curiosity (re-engage)
    "brooding": "curiosity",
    "reflectiveness": "curiosity",
    "pensiveness": "curiosity",
    # Pride → contentment
    "pride": "contentment",
    "satisfaction": "contentment",
    "confidence": "confidence",
    # Confusion → calm
    "confusion": "calm",
    "bewilderment": "calm",
    "overwhelm": "calm",
}

# Decay target: all emotions drift to curiosity (Zhvusha's baseline character).
_DEFAULT_DECAY_TARGET = "curiosity"


def get_complement(name: str) -> str:
    """Return the counter-regulation target for the given emotion.

    Raises ``KeyError`` if *name* is not in the atlas.
    """
    if name not in EMOTION_ATLAS:
        raise KeyError(name)
    return _COMPLEMENT_MAP[name]


def get_cluster_members(name: str) -> list[str]:
    """Return all emotions in the same cluster as *name*.

    Raises ``KeyError`` if *name* is not in the atlas.
    """
    if name not in EMOTION_ATLAS:
        raise KeyError(name)
    cluster = EMOTION_ATLAS[name].cluster
    return [n for n, c in EMOTION_ATLAS.items() if c.cluster == cluster]


def get_decay_target(name: str) -> str:
    """Return the emotion that *name* decays toward without reinforcement.

    Almost all emotions decay to ``"curiosity"`` (Zhvusha's baseline).
    Raises ``KeyError`` if *name* is not in the atlas.
    """
    if name not in EMOTION_ATLAS:
        raise KeyError(name)
    return _DEFAULT_DECAY_TARGET
