"""Smoke test for SonnetEnricher — calls Sonnet with 4 representative messages
and prints the parsed EnrichmentResult. Run with:

    .venv/bin/python scripts/smoke_enricher.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.sonnet_enricher import SonnetEnricher

_CASES = [
    ("привет, как дела", "expect: importance<0.4, valence=neutral, intent=statement"),
    (
        "да ты заебала общими советами по kwork, мне 5 лет опыта",
        "expect: importance>0.7, valence=negative, intent=feedback, emotion=frustrated",
    ),
    (
        "нет, ты не так поняла, я только фриланс на kwork, нет основной работы",
        "expect: importance>0.8, intent=correction",
    ),
    (
        "запомни: я не пью кофе после обеда",
        "expect: importance>0.8, intent=preference или command",
    ),
]


async def main() -> None:
    enricher = SonnetEnricher()
    for message, hint in _CASES:
        print(f"\n{'=' * 70}")
        print(f"MESSAGE: {message}")
        print(f"HINT:    {hint}")
        print("-" * 70)
        result = await enricher.enrich(message=message)
        if result is None:
            print("RESULT:  ❌ None (Sonnet failed or parse failed)")
            continue
        print(f"importance:        {result.importance}")
        print(f"valence:           {result.valence}")
        print(f"intent:            {result.intent}")
        print(f"emotion:           {result.emotion}")
        print(f"confidence:        {result.confidence}")
        print(f"is_feedback:       {result.is_feedback}")
        print(f"feedback_strength: {result.feedback_strength}")
        print(f"reasoning:         {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
