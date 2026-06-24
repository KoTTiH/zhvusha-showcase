"""Smoke test for Phase 2 LearningSignal extraction — calls Sonnet with
representative messages and prints learning_signal from each response.

Run with:

    .venv/bin/python scripts/smoke_learning_signal.py

What to look for:
  * рутинные сообщения → learning_signal = None
  * "не пиши формально" → rule, tone, apply_immediately=True, confidence>0.8
  * "нет, у меня только kwork" → correction, personal_facts, original_claim filled
  * "запомни: я не пью кофе" → fact or rule, apply_immediately=True
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.sonnet_enricher import SonnetEnricher

_CASES = [
    (
        "привет, как дела",
        "routine — expect learning_signal=None",
        None,  # no prev_bot_response
    ),
    (
        "не пиши мне так формально, мы на ты",
        "rule, tone, apply_immediately=True, confidence>0.8",
        "Здравствуйте! Я готова помочь вам с любым вопросом.",
    ),
    (
        "нет, у меня нет основной работы, я только на kwork 5 лет",
        "correction, personal_facts, original_claim filled",
        "Понимаю, что у вас есть основная работа помимо фриланса.",
    ),
    (
        "запомни: я не пью кофе после обеда",
        "fact or rule, apply_immediately=True",
        None,
    ),
    (
        "кажется мне больше нравятся краткие ответы, без воды",
        "preference, preferences, apply_immediately could be True or False",
        "Вот длинный развёрнутый ответ на ваш вопрос...",
    ),
    (
        "не обсуждай больше тему денег клиентов без моего разрешения",
        "boundary, boundaries, apply_immediately=True",
        None,
    ),
]


async def main() -> None:
    enricher = SonnetEnricher()
    for message, hint, prev in _CASES:
        print(f"\n{'=' * 75}")
        print(f"MESSAGE: {message}")
        print(f"HINT:    {hint}")
        print("-" * 75)
        result = await enricher.enrich(
            message=message,
            prev_bot_response=prev or "",
        )
        if result is None:
            print("RESULT:  ❌ None (Sonnet failed or parse failed)")
            continue

        print(f"importance:       {result.importance}")
        print(f"valence:          {result.valence}")
        print(f"intent:           {result.intent}")
        print(f"emotion:          {result.emotion}")
        print(f"confidence:       {result.confidence}")

        if result.learning_signal is None:
            print("learning_signal:  None")
        else:
            ls = result.learning_signal
            print("learning_signal:")
            print(f"  type:               {ls.type}")
            print(f"  statement:          {ls.statement}")
            print(f"  scope:              {ls.scope}")
            print(f"  confidence:         {ls.confidence}")
            print(f"  apply_immediately:  {ls.apply_immediately}")
            print(f"  original_claim:     {ls.original_claim}")

        print(f"reasoning:        {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
