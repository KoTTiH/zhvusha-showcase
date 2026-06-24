"""Dual-process decision making with metacognitive tracking.

System 1 (fast path): pattern matching via somatic markers, ~100ms
System 2 (slow path): full analytical LLM reasoning, ~3-10s
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.llm.protocols import LLMRequest
from src.memory import detect_domain

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.file_access import FileAccessService
    from src.llm.router import LLMRouter
    from src.memory import Episode
    from src.memory import EpisodicMemoryProtocol as EpisodicMemory
    from src.personality.protocols import PersonalityEvolutionProtocol
    from src.skills.base import AgentContext

logger = structlog.get_logger()

# Domain-based System 1 restrictions
FULL_AUTO_DOMAINS = frozenset({"chat"})
SUGGEST_ONLY_DOMAINS = frozenset({"kwork", "outreach", "content"})


@dataclass
class System1Result:
    similar_episode: Episode
    similarity: float
    valence: str
    confidence: float
    suggested_approach: str  # "act similar" | "avoid"


@dataclass
class DepthClassification:
    """Result of System 2 depth classification."""

    depth: str  # "QUICK" | "MEMORY" | "RESEARCH" | "UNCLEAR"


@dataclass
class System2Result:
    reasoning: str
    response: str
    system1_intuition: System1Result | None
    depth: str = "QUICK"  # depth classification used


@dataclass
class RetrievalPlan:
    """Output of unified planning step."""

    depth: str = "QUICK"  # QUICK | MEMORY | RESEARCH
    memory_queries: list[str] = field(default_factory=list)
    workspace_files: list[str] = field(default_factory=list)
    code_files: list[str] = field(default_factory=list)
    response: str = ""  # if QUICK — analyst tier can answer immediately


@dataclass
class RetrievalResult:
    """Combined context from memory + files."""

    memory_context: str = ""
    file_context: str = ""
    depth: str = "QUICK"
    quick_response: str = ""  # if set, skip second LLM call


@dataclass
class Decision:
    system: str  # "system1" | "system2"
    result: System1Result | System2Result
    confidence: float
    domain: str


_MD_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _strip_markdown_json(text: str) -> str:
    """Extract JSON from markdown code fences if present."""
    text = text.strip()
    m = _MD_JSON_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


class DecisionEngine:
    """Dual-process decision making."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        personality: PersonalityEvolutionProtocol,
        llm_router: LLMRouter,
        file_access: FileAccessService | None = None,
    ) -> None:
        self.episodic = episodic
        self.personality = personality
        self.llm = llm_router
        self._file_access = file_access
        self.metacognition = MetacognitionTracker()

    async def retrieve_for_question(
        self,
        question: str,
        recent_messages: str = "",
    ) -> RetrievalResult:
        """Unified planning + retrieval for chat_response skill."""
        if self._file_access is None:
            return RetrievalResult()

        plan = await self._plan_and_classify(question, recent_messages)

        if plan.depth == "QUICK" and plan.response:
            return RetrievalResult(depth="QUICK", quick_response=plan.response)

        return await self._execute_retrieval(plan)

    async def _plan_and_classify(
        self, question: str, recent_messages: str = ""
    ) -> RetrievalPlan:
        """Single analyst call: classify depth + plan retrieval."""
        from src.skills.chat_response.prompts import (
            GROUNDING_SECTION,
            PERSONALITY_ANCHOR,
        )

        workspace_index = (
            self._file_access.get_workspace_index() if self._file_access else ""
        )
        project_index = (
            self._file_access.get_project_index() if self._file_access else ""
        )

        personality_summary = self.personality.get_personality_tree_summary()

        # System prompt: personality + grounding (cacheable prefix)
        system = (
            f"Ты Жвуша.\n\n{PERSONALITY_ANCHOR}\n\n"
            f"{personality_summary}\n{GROUNDING_SECTION}"
        )

        history_block = ""
        if recent_messages:
            history_block = f"Недавние сообщения в чате:\n{recent_messages}\n\n"

        prompt = (
            f"{history_block}"
            f"Вопрос пользователя: {question}\n\n"
            f"Файлы в workspace:\n{workspace_index}\n\n"
            f"Файлы проекта:\n{project_index}\n\n"
            "Определи глубину ответа и план retrieval. Верни ТОЛЬКО JSON:\n"
            '{"depth": "QUICK|MEMORY|RESEARCH", '
            '"response": "ответ если QUICK, иначе пустая строка", '
            '"memory_queries": ["запрос1"], '
            '"workspace_files": ["path/to/file.md"], '
            '"code_files": []}\n\n'
            "ПРАВИЛА:\n"
            "- QUICK: простой вопрос, можешь ответить сразу. "
            "response = твой ответ, в характере Жвуши.\n"
            "- MEMORY: нужно покопаться в памяти и/или прочитать файлы.\n"
            "- RESEARCH: нужен глубокий анализ.\n"
            "- При depth=QUICK ответ чисто разговорный. "
            "Я знаю только ИМЕНА файлов — содержимое становится доступно "
            "после чтения через MEMORY/RESEARCH. Это архитектура, не забывчивость."
        )

        try:
            llm_response = await self.llm.generate(
                LLMRequest(
                    prompt=prompt,
                    system=system,
                    tier="analyst",
                    caller="decision_plan",
                )
            )
            raw = llm_response.text
            json_str = _strip_markdown_json(raw)
            data: dict[str, Any] = json.loads(json_str)
            return RetrievalPlan(
                depth=data.get("depth", "QUICK"),
                response=data.get("response", ""),
                memory_queries=data.get("memory_queries", []),
                workspace_files=data.get("workspace_files", []),
                code_files=data.get("code_files", []),
            )
        except Exception:
            logger.warning("plan_and_classify_failed", exc_info=True)
            return RetrievalPlan(depth="QUICK")

    async def _execute_retrieval(self, plan: RetrievalPlan) -> RetrievalResult:
        """Run memory queries + file reads from a retrieval plan."""
        memory_context = ""
        file_context = ""

        # Memory queries
        if plan.memory_queries:
            all_parts: list[str] = []
            seen: set[int] = set()
            for q in plan.memory_queries[:3]:
                episodes = await self.episodic.retrieve(q, limit=5)
                for ep in episodes:
                    if ep.id not in seen:
                        seen.add(ep.id)
                        all_parts.append(ep.content)
            memory_context = "\n---\n".join(all_parts)

        # File reads
        if self._file_access and (plan.workspace_files or plan.code_files):
            read_result = self._file_access.read_files(
                plan.workspace_files, plan.code_files
            )
            file_parts: list[str] = []
            now = datetime.now(UTC).isoformat()
            for path, content in read_result.workspace_contents.items():
                file_parts.append(
                    f'<FILE_CONTENT source="{path}" read_at="{now}">'
                    f"\n{content}\n</FILE_CONTENT>"
                )
            for path, content in read_result.code_contents.items():
                file_parts.append(
                    f'<FILE_CONTENT source="{path}" read_at="{now}">'
                    f"\n{content}\n</FILE_CONTENT>"
                )
            file_context = "\n\n".join(file_parts)

        return RetrievalResult(
            memory_context=memory_context,
            file_context=file_context,
            depth=plan.depth,
        )

    async def decide(
        self,
        situation: str,
        context: AgentContext,
    ) -> Decision:
        """Main entry point for decision making."""
        domain = self._detect_domain(situation, context)
        threshold = await self.metacognition.get_system1_threshold(domain)

        # Try System 1
        system1_result = await self._system1(situation, context, threshold)

        if system1_result is not None:
            if domain in FULL_AUTO_DOMAINS:
                # Fast path — System 1 generates response
                await self.metacognition.record_decision(
                    "system1", system1_result.confidence, domain
                )
                return Decision(
                    system="system1",
                    result=system1_result,
                    confidence=system1_result.confidence,
                    domain=domain,
                )
            # High-stakes domain — System 1 as intuition for System 2
            system2_result = await self._system2(situation, context, system1_result)
            await self.metacognition.record_decision(
                "system2", system1_result.confidence, domain
            )
            return Decision(
                system="system2",
                result=system2_result,
                confidence=system1_result.confidence,
                domain=domain,
            )

        # No System 1 match — full System 2
        system2_result = await self._system2(situation, context, None)
        await self.metacognition.record_decision("system2", 0.5, domain)
        return Decision(
            system="system2",
            result=system2_result,
            confidence=0.5,
            domain=domain,
        )

    def _detect_domain(self, situation: str, context: AgentContext) -> str:
        """Determine decision domain."""
        source = context.metadata.get("source", "")
        return detect_domain(situation, source=str(source), mode=context.mode)

    async def _system1(
        self,
        situation: str,
        context: AgentContext,
        threshold: float,
    ) -> System1Result | None:
        """Fast path: pattern matching with somatic markers."""
        results = await self.episodic.retrieve_by_somatic_marker(situation, limit=3)

        if not results:
            return None

        episode, similarity = results[0]

        # Arousal-based threshold bias: excited → slightly impulsive,
        # reflective → more cautious (per Anthropic emotion research).
        try:
            from src.personality import get_affective_state_manager

            affect = get_affective_state_manager().get_state()
            if affect.self_arousal > 0.7:
                threshold *= 0.9
            elif affect.self_emotion in ("brooding", "reflectiveness", "pensiveness"):
                threshold *= 1.1
        except Exception:  # noqa: S110
            pass  # affective state unavailable — use original threshold

        if similarity > 0.7 and episode.confidence > threshold:
            approach = "avoid" if episode.valence == "negative" else "act similar"
            return System1Result(
                similar_episode=episode,
                similarity=similarity,
                valence=episode.valence,
                confidence=episode.confidence,
                suggested_approach=approach,
            )

        return None

    async def classify_depth(self, question: str) -> DepthClassification:
        """Classify question depth: QUICK / MEMORY / RESEARCH / UNCLEAR."""
        from src.personality import PERSONALITY_COMPACT

        prompt = (
            f"Пользователь спросил: {question}\n\n"
            "Как лучше ответить? Выбери ОДИН вариант:\n"
            "A) QUICK — простой вопрос, достаточно текущего контекста\n"
            "B) MEMORY — нужно покопаться в памяти и знаниях\n"
            "C) RESEARCH — нужен поиск в интернете + память\n"
            "D) UNCLEAR — не понятно насколько глубоко копать\n\n"
            "Верни ТОЛЬКО букву: A, B, C или D"
        )

        try:
            llm_response = await self.llm.generate(
                LLMRequest(
                    prompt=prompt,
                    system=PERSONALITY_COMPACT,
                    tier="analyst",
                    caller="decision_depth",
                )
            )
            letter = llm_response.text.strip().upper()[:1]
            depth_map = {"A": "QUICK", "B": "MEMORY", "C": "RESEARCH", "D": "UNCLEAR"}
            depth = depth_map.get(letter, "QUICK")
        except Exception:
            logger.warning("depth_classification_failed", exc_info=True)
            depth = "QUICK"

        return DepthClassification(depth=depth)

    async def _active_retrieval(self, question: str) -> str:
        """Multi-step memory retrieval for MEMORY-depth questions."""
        from src.personality import PERSONALITY_COMPACT

        plan_prompt = (
            f"Вопрос: {question}\n"
            "Какие поисковые запросы по памяти нужны?\n"
            "Верни JSON с полем queries — список объектов с полями "
            "query (строка) и source_filter (список строк)."
        )

        try:
            plan_llm_response = await self.llm.generate(
                LLMRequest(
                    prompt=plan_prompt,
                    system=PERSONALITY_COMPACT,
                    tier="analyst",
                    caller="decision_retrieval",
                )
            )
            plan = json.loads(plan_llm_response.text)
            queries = plan.get("queries", [])
        except Exception:
            logger.warning("active_retrieval_plan_failed", exc_info=True)
            queries = [{"query": question, "source_filter": None}]

        all_context: list[str] = []
        seen_ids: set[int] = set()

        for q in queries[:3]:
            query_text = q.get("query", question)
            source_filter = q.get("source_filter")

            episodes = await self.episodic.retrieve(
                query_text,
                limit=5,
                source_filter=source_filter,
            )

            for ep in episodes:
                if ep.id in seen_ids:
                    continue
                seen_ids.add(ep.id)
                all_context.append(ep.content)

                if ep.metadata_json:
                    try:
                        meta = json.loads(ep.metadata_json)
                        file_path = meta.get("file_path")
                        if file_path:
                            all_context.append(f"[Knowledge: {file_path}]")
                    except json.JSONDecodeError:
                        pass

        return "\n---\n".join(all_context) if all_context else ""

    async def _system2(
        self,
        situation: str,
        context: AgentContext,
        system1_result: System1Result | None,
    ) -> System2Result:
        """Slow path: full analytical reasoning with depth classification."""
        classification = await self.classify_depth(situation)
        depth = classification.depth

        extra_context = ""
        if depth in ("MEMORY", "RESEARCH"):
            extra_context = await self._active_retrieval(situation)

        from src.skills.chat_response.prompts import (
            GROUNDING_SECTION,
            PERSONALITY_ANCHOR,
        )

        personality_summary = self.personality.get_personality_tree_summary()
        system_prompt = (
            f"Ты Жвуша.\n\n{PERSONALITY_ANCHOR}\n\n"
            f"{personality_summary}\n{GROUNDING_SECTION}"
        )

        user_prompt = situation
        if system1_result is not None:
            user_prompt = (
                f"Моя первая интуиция говорит: {system1_result.suggested_approach} "
                f"(на основе похожего прошлого опыта с исходом "
                f"{system1_result.valence}, "
                f"уверенность {system1_result.confidence:.1f}). "
                f"Но уверенности мало, подумаю тщательнее.\n\n"
                f"{situation}"
            )

        if extra_context:
            user_prompt += f"\n\n<MEMORY_FACTS>\n{extra_context}\n</MEMORY_FACTS>"

        if depth == "UNCLEAR":
            user_prompt += (
                "\n\nУточни у пользователя: могу ответить коротко из того что знаю, "
                "или покопаться глубже — поискать по своим заметкам, видео, каналам."
            )

        llm_response = await self.llm.generate(
            LLMRequest(
                prompt=user_prompt,
                system=system_prompt,
                tier="analyst",
                caller="decision_system2",
            )
        )

        return System2Result(
            reasoning="Full analytical reasoning via LLM",
            response=llm_response.text,
            system1_intuition=system1_result,
            depth=depth,
        )

    async def record_outcome(
        self,
        episode_id: int,
        outcome: str,
        domain: str,
    ) -> None:
        """Record decision outcome for learning."""
        valence = "positive" if outcome == "positive" else "negative"
        confidence = 0.8

        await self.episodic.update_valence(episode_id, valence, confidence)
        await self.metacognition.record_outcome(
            domain, "system1", outcome == "positive"
        )


class MetacognitionTracker:
    """Tracks decision quality over time.

    Per-domain tracking with dynamic System 1 thresholds.
    """

    def __init__(self, stats_path: Path | None = None) -> None:
        self._stats_path = stats_path
        self._stats: dict[str, dict[str, int]] = {}
        if stats_path is not None and stats_path.exists():
            try:
                self._stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._stats = {}

    def _get_domain_stats(self, domain: str) -> dict[str, int]:
        if domain not in self._stats:
            self._stats[domain] = {
                "system1_decisions": 0,
                "system2_decisions": 0,
                "system1_good": 0,
                "system1_bad": 0,
                "total_decisions": 0,
            }
        return self._stats[domain]

    async def record_decision(
        self,
        system_used: str,
        confidence: float,
        domain: str,
    ) -> None:
        stats = self._get_domain_stats(domain)
        if system_used == "system1":
            stats["system1_decisions"] += 1
        else:
            stats["system2_decisions"] += 1
        stats["total_decisions"] += 1
        self._save()

    async def record_outcome(
        self,
        domain: str,
        system_used: str,
        was_correct: bool,
    ) -> None:
        stats = self._get_domain_stats(domain)
        if was_correct:
            stats["system1_good"] += 1
        else:
            stats["system1_bad"] += 1
        self._save()

    async def get_system1_threshold(self, domain: str) -> float:
        """Dynamic confidence threshold for System 1.

        Default: 0.7
        After 3+ good outcomes → 0.6
        After 2+ bad outcomes → 0.8
        Clamp to [0.5, 0.9], stabilize after 20+ decisions.
        """
        stats = self._get_domain_stats(domain)
        good = stats.get("system1_good", 0)
        bad = stats.get("system1_bad", 0)
        total = stats.get("total_decisions", 0)

        threshold = 0.7

        if good >= 3:
            threshold -= 0.1
        if bad >= 2:
            threshold += 0.1

        # Stabilization after 20+ decisions
        if total >= 20:
            threshold = 0.7 + 0.1 * (bad - good) / max(total, 1)

        return max(0.5, min(0.9, threshold))

    async def should_suggest_strategy_change(
        self,
        domain: str,
    ) -> str | None:
        """Suggest strategy change after 3+ recent failures."""
        stats = self._get_domain_stats(domain)
        if stats.get("system1_bad", 0) >= 3:
            return (
                f"Domain '{domain}': {stats['system1_bad']} recent negative outcomes. "
                f"Consider adjusting approach."
            )
        return None

    def _save(self) -> None:
        if self._stats_path is not None:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_path.write_text(
                json.dumps(self._stats, indent=2), encoding="utf-8"
            )
