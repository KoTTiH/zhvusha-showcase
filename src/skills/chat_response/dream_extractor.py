"""Fire-and-forget Sonnet-based dream detection in Zhvusha's responses.

After Zhvusha responds in personal mode, DreamExtractor checks if the
response contains an idea, wish, or dream about a new capability, project,
or goal. On success returns DreamResult; on any failure returns None
(never raises).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from src.llm.protocols import LLMRequest
from src.llm.router import get_router
from src.memory import parse_enrichment_json
from src.personality import PERSONALITY_COMPACT

if TYPE_CHECKING:
    from src.core.config import Tier

logger = structlog.get_logger()


class DreamResult(BaseModel):
    """Structured output from dream detection."""

    has_dream: bool
    dream_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


_DREAM_SYSTEM_PROMPT = f"""\
{PERSONALITY_COMPACT}

Подходи с тёплым интересом к собственным мечтам, но будь честной — \
не выдавай рутинные мысли за мечты.

Проанализируй СВОЙ ответ (не сообщение Никиты). Есть ли в нём идея, \
желание или мечта о:
- Новой способности, которую хочешь получить
- Проекте, который хочешь начать
- Цели, к которой хочешь стремиться
- Навыке, который хочешь освоить

НЕ считать мечтой:
- Рутинные ответы, пересказ фактов
- Описание уже реализованных функций
- Предложения для Никиты (это его мечты, не твои)
- Общие фразы ("было бы круто")
- Мечту, которая уже записана в <EXISTING_DREAMS> (даже если \
сформулирована иначе — если смысл тот же, это дубликат)

Если мечт несколько — выбери ОДНУ самую важную.

Ответь СТРОГИМ JSON:
{{"has_dream": bool, "dream_text": "краткая формулировка мечты", "confidence": 0.0-1.0}}

Без markdown-блоков. Только валидный JSON."""


class DreamExtractor:
    """Detects dreams/wishes in Zhvusha's responses via LLM."""

    def __init__(self, *, tier: Tier = "worker") -> None:
        self._tier = tier

    async def check(
        self,
        bot_response: str,
        recent_context: str = "",
        existing_dreams: str = "",
    ) -> DreamResult | None:
        """Analyse bot response for dreams. Never raises — returns None on failure."""
        prompt = self._build_prompt(bot_response, recent_context, existing_dreams)

        try:
            router = get_router()
            llm_response = await router.generate(
                LLMRequest(
                    prompt=prompt,
                    system=_DREAM_SYSTEM_PROMPT,
                    tier=self._tier,
                    temperature=0.0,
                    caller="dream_extract",
                )
            )
            raw = llm_response.text
        except Exception:
            logger.warning("dream_extractor_llm_failed", exc_info=True)
            return None

        parsed = parse_enrichment_json(raw)
        if parsed is None:
            logger.warning(
                "dream_extractor_parse_failed",
                raw_sample=raw[:200] if raw else None,
            )
            return None

        try:
            return DreamResult(**parsed)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            logger.warning(
                "dream_extractor_schema_mismatch",
                raw_sample=raw[:200] if raw else None,
            )
            return None

    @staticmethod
    def _build_prompt(
        bot_response: str, recent_context: str, existing_dreams: str = ""
    ) -> str:
        parts: list[str] = []
        if recent_context:
            parts.append(
                f"<RECENT_CONVERSATION>\n{recent_context}\n</RECENT_CONVERSATION>"
            )
        parts.append(f"<MY_RESPONSE>\n{bot_response}\n</MY_RESPONSE>")
        if existing_dreams.strip():
            parts.append(f"<EXISTING_DREAMS>\n{existing_dreams}\n</EXISTING_DREAMS>")
        parts.append(
            "Проанализируй свой ответ. Есть ли мечта? "
            "Ответь строгим JSON по схеме DreamResult."
        )
        return "\n\n".join(parts)


_extractor: DreamExtractor | None = None


def get_dream_extractor() -> DreamExtractor:
    """Singleton accessor for DreamExtractor."""
    global _extractor
    if _extractor is None:
        from src.core.config import get_settings

        _extractor = DreamExtractor(
            tier=get_settings().dream_extraction_tier,
        )
    return _extractor
