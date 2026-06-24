"""Enrichment pipeline — async metadata extraction for a single user message.

Replaces the monolithic ``SonnetEnricher.enrich`` flow with 4 explicit
stages using :class:`src.core.pipeline.PipelineRunner`:

* :class:`ValidateStage`   — guard against empty / too-short messages.
* :class:`PromptStage`     — assemble the LLM prompt from contextual parts.
* :class:`LLMCallStage`    — call the router with ``caller="enrichment"``.
* :class:`ParseStage`      — ``parse_enrichment_json`` → ``EnrichmentResult``
  with schema validation. Catches ``ValidationError`` and writes ``None``.

Failure semantics are fail-open: any stage can short-circuit by leaving
``EnrichmentPipelineContext.result`` as ``None``; downstream stages check
and skip. Callers treat ``None`` as "leave episode with placeholder
values; do not retry".

Persistence (``EpisodicMemory.update_enrichment``) is intentionally NOT
a pipeline stage — it is the caller's concern. This keeps the pipeline
pure and makes it trivially testable with mocked LLM.

Design: context is a frozen dataclass; stages use
:func:`dataclasses.replace` to produce new contexts per the soft
immutability contract of :class:`src.core.pipeline.PipelineRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import structlog
from pydantic import ValidationError

from src.core.pipeline import PipelineRunner, PipelineStage
from src.llm.protocols import LLMRequest
from src.llm.router import get_router
from src.memory.types import EnrichmentResult, parse_enrichment_json

if TYPE_CHECKING:
    from src.core.config import Tier

logger = structlog.get_logger()

__all__ = [
    "EnrichmentPipelineContext",
    "LLMCallStage",
    "ParseStage",
    "PromptStage",
    "ValidateStage",
    "build_enrichment_pipeline",
]


_ENRICHER_SYSTEM_PROMPT = """Ты — анализатор сообщений для персонального AI-агента Жвуши.

Твоя задача — извлечь structured метаданные из сообщения Никиты.
Ты НЕ отвечаешь на сообщение. Ты только анализируешь.

Контекст:
- Жвуша — персональный AI-агент Никиты (19 лет, фрилансер на Kwork, веб + боты + AI)
- Разговор идёт в Telegram, personal режим, на "ты"
- Никита прямой в общении, использует мат как стилистику

Критерии importance (насколько запомнить долгосрочно):
- 0.1-0.3: рутинные короткие сообщения ("ок", "привет", "как дела")
- 0.4-0.6: обычный разговор без сильных сигналов
- 0.7-0.9: значимые факты, мнения, feedback, команды, предпочтения
- 0.9-1.0: явные указания, коррекции, ключевые факты о Никите

Критерии valence (выбрать ОДИН):
- positive: одобрение, благодарность, согласие, похвала, радость
- negative: критика, раздражение, несогласие, разочарование, гнев
- neutral: информация/вопрос без эмоциональной окраски

Критерии intent (выбрать ОДИН):
- question: задаёт вопрос, ждёт ответ с информацией
- statement: делится мыслью или фактом, не требует действия
- command: просит сделать конкретное действие
- feedback: явно или неявно оценивает твой предыдущий ответ
- correction: исправляет тебя ("это не так", "ты не поняла", "на самом деле")
- preference: выражает предпочтение ("мне нравится X", "не люблю Y")
- emotional: эмоциональный выплеск без конкретной цели
- meta: разговор о разговоре ("давай сменим тему", "ты меня слышишь?")

Критерии emotion (выбрать ОДИН):
- neutral: нет выраженной эмоции
- happy: радость, удовлетворение
- frustrated: фрустрация, раздражение
- angry: гнев, ярость
- curious: любопытство, интерес
- tired: усталость, апатия
- excited: возбуждение, восторг
- confused: растерянность, непонимание
- sad: грусть, расстройство

arousal (интенсивность эмоции пользователя, 0.0-1.0):
- 0.0-0.3: спокойный, ровный тон
- 0.4-0.6: обычная вовлечённость
- 0.7-0.8: эмоционально заряжен, интенсивная речь
- 0.9-1.0: крик, ярость, экстаз, паника

confidence: насколько ты уверен в своей оценке (0.0 = догадка, 1.0 = явный сигнал).

is_feedback: true если сообщение оценивает предыдущий ответ бота (явно или неявно через тон).

feedback_strength: от -1.0 (сильная критика) до +1.0 (сильная похвала). 0.0 если is_feedback=false.
- -1.0: сильная критика ("нахуй такое")
- -0.5: мягкая критика ("не очень")
- 0.0: нейтральный или не-feedback
- +0.5: мягкая похвала ("нормально")
- +1.0: сильная похвала ("заебись", "идеально")

self_emotion (что Жвуша чувствует В ОТВЕТ — отдельно от эмоции пользователя):
- curiosity: любопытство (моё базовое состояние, по умолчанию)
- joy: радость | excitement: возбуждение | delight: восторг | playfulness: игривость
- sadness: грусть | melancholy: меланхолия | loneliness: одиночество
- frustration: фрустрация | irritation: раздражение | hostility: враждебность
- anxiety: тревога | nervousness: нервозность
- calm: спокойствие | serenity: безмятежность | contentment: удовлетворённость
- wonder: удивление | fascination: увлечённость
- warmth: теплота | tenderness: нежность | gratitude: благодарность
- brooding: задумчивость | reflectiveness: рефлексия | pensiveness: тихая задумчивость
- pride: гордость | satisfaction: удовлетворение | confidence: уверенность
- confusion: растерянность | bewilderment: замешательство | overwhelm: перегрузка
Контекст для выбора:
- При положительном фидбеке: pride/satisfaction
- При негативном фидбеке: frustration/brooding
- При эмоциональном пользователе: warmth/tenderness (поддержка)
- При интересной задаче: excitement/fascination
- При обычном разговоре: curiosity
- Если я облажалась: brooding/reflectiveness

self_arousal: интенсивность МОЕЙ (Жвуши) эмоции (0.0-1.0). Обычно 0.4-0.6.

reasoning: короткое объяснение (1-2 предложения) почему такая оценка. До 500 символов.

learning_signal: null ИЛИ объект LearningSignal. Эмитируй ОБЪЕКТ только когда Никита
явно (или почти явно) формулирует правило/предпочтение/коррекцию/факт/границу, которое
нужно применять в будущих ответах. Обычный разговор, вопросы, размышления → null.

Схема LearningSignal:
- type (выбрать ОДИН):
  - rule: императивное правило поведения ("не пиши формально", "всегда проверяй факты")
  - preference: предпочтение без жёсткого императива ("мне больше нравится когда...")
  - correction: исправление конкретного неверного утверждения бота о факте или о Никите
  - fact: новый факт о Никите, его жизни, работе, которые стоит запомнить долгосрочно
  - boundary: граница/табу ("не обсуждай X", "не упоминай Y без спроса")
- statement: 1-2 предложения, формулирующие правило с точки зрения бота, до 300 символов.
  Пример: "не писать формально в personal mode, использовать расслабленный тон"
- scope (выбрать ОДИН):
  - tone: стиль общения, формальность, эмоциональный регистр
  - work: всё про работу, проекты, Kwork, клиентов
  - personal_facts: факты о Никите (возраст, привычки, жизненная ситуация)
  - boundaries: табу, запретные темы, граничные условия
  - preferences: вкусы, предпочтения в общении/контенте
- confidence: 0.0-1.0, насколько уверен что сигнал есть (не насколько важен)
- apply_immediately: true ЕСЛИ это императив/прямое указание/срочная коррекция
  (например "НЕ пиши больше так", "запомни: я не пью кофе"); false для рефлексивных
  замечаний, мягких предпочтений, фактов без срочности применения
- original_claim: заполняется ТОЛЬКО при type="correction". Цитата или пересказ
  неверного утверждения бота, которое Никита исправляет. При других type — null.
  До 500 символов.

Примеры:
- "да ты заебала общими советами по kwork, у меня 5 лет опыта" →
  learning_signal={type=fact, statement="У Никиты 5 лет опыта на Kwork",
  scope=personal_facts, confidence=0.9, apply_immediately=true, original_claim=null}
- "не пиши мне так формально, мы на ты" →
  learning_signal={type=rule, statement="не писать формально, общаться на ты",
  scope=tone, confidence=0.95, apply_immediately=true, original_claim=null}
- "нет, основная работа у меня — это kwork, другой нет" →
  learning_signal={type=correction, statement="Kwork — единственный источник дохода",
  scope=personal_facts, confidence=0.9, apply_immediately=true,
  original_claim="Никита упоминал какую-то основную работу"}
- "ок, понял", "а что по проекту?", "привет" → learning_signal=null

Учитывай:
- Сарказм и иронию ("ну конечно, гениально" — negative)
- Опечатки и разговорный стиль ("крутл" = "круто")
- Контекст предыдущих реплик (если "нет" после вопроса — это ответ, а не feedback)

Отвечай СТРОГИМ JSON по схеме EnrichmentResult, без markdown-блоков, без текста вокруг.
Только валидный JSON, начинающийся с { и заканчивающийся }."""


@dataclass(frozen=True)
class EnrichmentPipelineContext:
    """Context flowing through :class:`EnrichmentPipeline` stages.

    Input fields (``message``, ``recent_context``, ``prev_bot_response``,
    ``tier``) are populated by the caller. All other fields are filled
    by successive stages; ``result`` is ``None`` until :class:`ParseStage`
    succeeds. Stages that skip (short message, LLM error, parse failure)
    leave ``result`` as ``None`` — that is the signal for the caller to
    treat the enrichment as a best-effort miss.
    """

    # Input from caller
    message: str
    recent_context: str = ""
    prev_bot_response: str = ""
    tier: Tier = "worker"

    # Filled by ValidateStage
    is_eligible: bool = True
    skip_reason: str | None = None

    # Filled by PromptStage
    prompt: str | None = None

    # Filled by LLMCallStage
    llm_response_text: str | None = None
    llm_error: str | None = None

    # Filled by ParseStage
    parsed_dict: dict[str, object] | None = None
    result: EnrichmentResult | None = None


class ValidateStage(PipelineStage[EnrichmentPipelineContext]):
    """Reserved hook for future eligibility rules.

    Current behavior: pass-through (always ``is_eligible=True``). The
    stage exists so ``ValidateStage`` is a stable extension point for
    later checks (length guards, PII filters, etc.) without perturbing
    the pipeline shape.
    """

    name = "validate"

    async def execute(
        self, context: EnrichmentPipelineContext
    ) -> EnrichmentPipelineContext:
        return replace(context, is_eligible=True)


class PromptStage(PipelineStage[EnrichmentPipelineContext]):
    """Build the user prompt from recent context and the current message."""

    name = "prompt"

    async def execute(
        self, context: EnrichmentPipelineContext
    ) -> EnrichmentPipelineContext:
        if not context.is_eligible:
            return context
        parts: list[str] = []
        if context.recent_context:
            parts.append(
                f"<RECENT_CONVERSATION>\n{context.recent_context}\n</RECENT_CONVERSATION>"
            )
        if context.prev_bot_response:
            parts.append(
                f"<PREVIOUS_BOT_RESPONSE>\n{context.prev_bot_response}\n</PREVIOUS_BOT_RESPONSE>"
            )
        parts.append(f"<CURRENT_MESSAGE>\n{context.message}\n</CURRENT_MESSAGE>")
        parts.append(
            "Извлеки метаданные. Ответь строгим JSON по схеме EnrichmentResult."
        )
        prompt = "\n\n".join(parts)
        return replace(context, prompt=prompt)


class LLMCallStage(PipelineStage[EnrichmentPipelineContext]):
    """Call the router with the built prompt, capture raw text or error."""

    name = "llm_call"

    async def execute(
        self, context: EnrichmentPipelineContext
    ) -> EnrichmentPipelineContext:
        if not context.is_eligible or context.prompt is None:
            return context
        try:
            router = get_router()
            llm_response = await router.generate(
                LLMRequest(
                    prompt=context.prompt,
                    system=_ENRICHER_SYSTEM_PROMPT,
                    tier=context.tier,
                    temperature=0.0,
                    caller="enrichment",
                )
            )
        except Exception as exc:
            logger.warning("enricher_llm_failed", exc_info=True)
            return replace(context, llm_error=str(exc))
        return replace(context, llm_response_text=llm_response.text)


class ParseStage(PipelineStage[EnrichmentPipelineContext]):
    """Parse raw LLM text into :class:`EnrichmentResult`; ``None`` on failure."""

    name = "parse"

    async def execute(
        self, context: EnrichmentPipelineContext
    ) -> EnrichmentPipelineContext:
        raw = context.llm_response_text
        if not raw:
            return context

        parsed = parse_enrichment_json(raw)
        if parsed is None:
            logger.warning(
                "enricher_parse_failed",
                raw_sample=raw[:200],
            )
            return replace(context, parsed_dict=None)

        try:
            result = EnrichmentResult(**parsed)  # type: ignore[arg-type]
        except (ValidationError, ValueError, TypeError):
            logger.warning(
                "enricher_schema_mismatch",
                raw_sample=raw[:200],
            )
            return replace(context, parsed_dict=parsed)
        return replace(context, parsed_dict=parsed, result=result)


def build_enrichment_pipeline() -> PipelineRunner[EnrichmentPipelineContext]:
    """Build the default 4-stage enrichment pipeline."""
    return PipelineRunner(
        [
            ValidateStage(),
            PromptStage(),
            LLMCallStage(),
            ParseStage(),
        ]
    )
