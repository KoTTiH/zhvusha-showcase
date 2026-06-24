"""Sleep-Time Agent — background knowledge maintenance during idle periods.

Runs on cheap models (Haiku/worker tier). Applies tags, summaries, and
categories directly to knowledge entries.

Throttling:
  * minimum cooldown between cycles (``_MIN_CYCLE_INTERVAL``)
  * adaptive backoff based on recent LLM call rate
  * batch LLM call per task with per-entry sequential fallback
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from src.knowledge import KnowledgeStore
    from src.llm.router import LLMRouter
    from src.monitoring.usage_tracker import UsageTracker

logger = structlog.get_logger()

# Minimum interval between maintenance cycles (seconds)
_MIN_CYCLE_INTERVAL = 300.0  # 5 minutes
# Adaptive backoff thresholds — see KB #90 part 3
_BACKOFF_THRESHOLD = 40  # > this: double the cooldown
_FULL_STOP_THRESHOLD = 50  # > this: cooldown = 1 hour (full stop)
_FULL_STOP_COOLDOWN = 3600.0  # 1 hour
# Maximum entries per task per cycle — one batched LLM call processes them all
_MAX_ENTRIES_PER_TASK = 15


class _BatchFailedError(RuntimeError):
    """Raised when batch parsing fails and the task should fall back."""


class SleepTimeAgent:
    """Runs background knowledge maintenance when the daemon is idle."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        llm_router: LLMRouter,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self._store = knowledge_store
        self._llm = llm_router
        self._usage_tracker = usage_tracker
        self._last_cycle_ts: float = 0.0

    async def run_maintenance_cycle(self) -> int:
        """Run one cycle of maintenance tasks. Returns count of proposals created.

        Enforces adaptive cooldown:
          * default 5 minutes between cycles
          * doubles to 10 minutes when > ``_BACKOFF_THRESHOLD`` calls/hour
          * stretches to 1 hour when > ``_FULL_STOP_THRESHOLD`` calls/hour
        """
        now = time.monotonic()
        elapsed = now - self._last_cycle_ts
        required = self._required_cooldown()
        if self._last_cycle_ts > 0.0 and elapsed < required:
            logger.debug(
                "sleep_agent_cooldown",
                remaining=int(required - elapsed),
                required=int(required),
            )
            return 0

        self._last_cycle_ts = now

        total = 0
        for task_fn in [
            self._tag_untagged,
            self._summarize_unsummarized,
            self._categorize_uncategorized,
        ]:
            try:
                count = await task_fn()
                total += count
            except Exception:
                logger.warning(
                    "sleep_agent_task_failed",
                    task=task_fn.__name__,
                    exc_info=True,
                )
        return total

    def _required_cooldown(self) -> float:
        """Return required seconds before the next cycle may start.

        Fail-open: any tracker error falls back to the default cooldown.
        """
        if self._usage_tracker is None:
            return _MIN_CYCLE_INTERVAL
        try:
            calls = self._usage_tracker.get_calls_in_last_hour()
        except Exception:
            logger.warning("sleep_agent_tracker_error", exc_info=True)
            return _MIN_CYCLE_INTERVAL
        if calls > _FULL_STOP_THRESHOLD:
            logger.warning("sleep_agent_full_stop", hourly_calls=calls)
            return _FULL_STOP_COOLDOWN
        if calls > _BACKOFF_THRESHOLD:
            logger.warning("sleep_agent_backoff", hourly_calls=calls)
            return _MIN_CYCLE_INTERVAL * 2
        return _MIN_CYCLE_INTERVAL

    # ------------------------------------------------------------------ tag

    async def _tag_untagged(self) -> int:
        entries = await self._store.get_untagged(limit=_MAX_ENTRIES_PER_TASK)
        if not entries:
            return 0
        try:
            tags_by_id = await self._batch_tag(entries)
        except _BatchFailedError:
            return await self._sequential_tag(entries)

        count = 0
        for entry in entries:
            tags = tags_by_id.get(entry.id, [])
            if not tags:
                continue
            try:
                await self._store.update_entry(entry.id, tags=tags)
                count += 1
                logger.info("sleep_tagged", entry_id=entry.id, tags=tags)
            except Exception:
                logger.warning(
                    "sleep_tag_update_failed", entry_id=entry.id, exc_info=True
                )
        return count

    async def _batch_tag(self, entries: list[Any]) -> dict[int, list[str]]:
        prompt = _build_tag_batch_prompt(entries)
        result = await self._llm.generate(
            LLMRequest(prompt=prompt, tier="worker", caller="sleep_agent_batch")
        )
        return _parse_tag_batch(result.text, entries)

    async def _sequential_tag(self, entries: list[Any]) -> int:
        """Fallback: per-entry LLM calls when batch JSON parsing failed."""
        count = 0
        for entry in entries:
            try:
                llm_result = await self._llm.generate(
                    LLMRequest(
                        prompt=(
                            f"Предложи 3-5 тегов для записи:\n"
                            f"Заголовок: {entry.title}\n"
                            f"Содержимое: {entry.content[:500]}\n\n"
                            f"Ответь ТОЛЬКО через запятую, без пояснений."
                        ),
                        tier="worker",
                        caller="sleep_agent",
                    )
                )
                tags = [t.strip() for t in llm_result.text.split(",") if t.strip()]
                if tags:
                    await self._store.update_entry(entry.id, tags=tags)
                    count += 1
                    logger.info("sleep_tagged_fallback", entry_id=entry.id, tags=tags)
            except Exception:
                logger.warning("sleep_tag_failed", entry_id=entry.id, exc_info=True)
        return count

    # ------------------------------------------------------------------ summary

    async def _summarize_unsummarized(self) -> int:
        entries = await self._store.get_unsummarized(limit=_MAX_ENTRIES_PER_TASK)
        if not entries:
            return 0
        try:
            summary_by_id = await self._batch_summarize(entries)
        except _BatchFailedError:
            return await self._sequential_summarize(entries)

        count = 0
        for entry in entries:
            summary = summary_by_id.get(entry.id, "").strip()
            if not summary:
                continue
            try:
                await self._store.update_entry(entry.id, summary=summary)
                count += 1
                logger.info("sleep_summarized", entry_id=entry.id)
            except Exception:
                logger.warning(
                    "sleep_summarize_update_failed", entry_id=entry.id, exc_info=True
                )
        return count

    async def _batch_summarize(self, entries: list[Any]) -> dict[int, str]:
        prompt = _build_summary_batch_prompt(entries)
        result = await self._llm.generate(
            LLMRequest(prompt=prompt, tier="worker", caller="sleep_agent_batch")
        )
        return _parse_summary_batch(result.text, entries)

    async def _sequential_summarize(self, entries: list[Any]) -> int:
        count = 0
        for entry in entries:
            try:
                llm_result = await self._llm.generate(
                    LLMRequest(
                        prompt=(
                            f"Напиши краткое саммари (2-3 предложения) для:\n"
                            f"Заголовок: {entry.title}\n"
                            f"Содержимое: {entry.content[:1000]}\n\n"
                            f"Только саммари, без пояснений."
                        ),
                        tier="worker",
                        caller="sleep_agent",
                    )
                )
                summary = llm_result.text.strip()
                if summary:
                    await self._store.update_entry(entry.id, summary=summary)
                    count += 1
                    logger.info("sleep_summarized_fallback", entry_id=entry.id)
            except Exception:
                logger.warning(
                    "sleep_summarize_failed", entry_id=entry.id, exc_info=True
                )
        return count

    # ------------------------------------------------------------------ category

    async def _categorize_uncategorized(self) -> int:
        entries = await self._store.get_uncategorized(limit=_MAX_ENTRIES_PER_TASK)
        if not entries:
            return 0

        categories = await self._store.browse_categories()
        cat_names = (
            ", ".join(c.path for c in categories) if categories else "нет категорий"
        )

        try:
            category_by_id = await self._batch_categorize(entries, cat_names)
        except _BatchFailedError:
            return await self._sequential_categorize(entries, cat_names)

        count = 0
        for entry in entries:
            raw = category_by_id.get(entry.id, "").strip()
            if not raw:
                continue
            category_path = raw.lower().replace(" ", "_")
            name_ru = " > ".join(p.capitalize() for p in category_path.split("."))
            try:
                cat_id = await self._store.get_or_create_category(
                    category_path, name_ru
                )
                await self._store.update_entry(entry.id, category_id=cat_id)
                count += 1
                logger.info(
                    "sleep_categorized", entry_id=entry.id, category=category_path
                )
            except Exception:
                logger.warning(
                    "sleep_categorize_update_failed",
                    entry_id=entry.id,
                    exc_info=True,
                )
        return count

    async def _batch_categorize(
        self, entries: list[Any], cat_names: str
    ) -> dict[int, str]:
        prompt = _build_category_batch_prompt(entries, cat_names)
        result = await self._llm.generate(
            LLMRequest(prompt=prompt, tier="worker", caller="sleep_agent_batch")
        )
        return _parse_category_batch(result.text, entries)

    async def _sequential_categorize(self, entries: list[Any], cat_names: str) -> int:
        count = 0
        for entry in entries:
            try:
                llm_result = await self._llm.generate(
                    LLMRequest(
                        prompt=(
                            f"К какой категории относится запись?\n"
                            f"Заголовок: {entry.title}\n"
                            f"Содержимое: {entry.content[:300]}\n\n"
                            f"Существующие категории: {cat_names}\n"
                            f"Ответь ОДНИМ словом — путь категории (например: tools.python). "
                            f"Можно предложить новую."
                        ),
                        tier="worker",
                        caller="sleep_agent",
                    )
                )
                category_path = llm_result.text.strip().lower().replace(" ", "_")
                if category_path:
                    name_ru = " > ".join(
                        p.capitalize() for p in category_path.split(".")
                    )
                    cat_id = await self._store.get_or_create_category(
                        category_path, name_ru
                    )
                    await self._store.update_entry(entry.id, category_id=cat_id)
                    count += 1
                    logger.info(
                        "sleep_categorized_fallback",
                        entry_id=entry.id,
                        category=category_path,
                    )
            except Exception:
                logger.warning(
                    "sleep_categorize_failed", entry_id=entry.id, exc_info=True
                )
        return count


# ---------------------------------------------------------------------------
# Batch prompt / parser helpers (module-level for easier unit testing)


def _build_tag_batch_prompt(entries: list[Any]) -> str:
    rows = "\n".join(
        f'  {{"id": {e.id}, "title": {json.dumps(e.title, ensure_ascii=False)}, '
        f'"content": {json.dumps(e.content[:500], ensure_ascii=False)}}}'
        for e in entries
    )
    return (
        "Для каждой записи ниже предложи 3-5 тегов.\n"
        "Верни ТОЛЬКО JSON-массив без пояснений, формат: "
        '[{"id": <int>, "tags": ["tag1", ...]}, ...].\n\n'
        f"Записи:\n[\n{rows}\n]"
    )


def _build_summary_batch_prompt(entries: list[Any]) -> str:
    rows = "\n".join(
        f'  {{"id": {e.id}, "title": {json.dumps(e.title, ensure_ascii=False)}, '
        f'"content": {json.dumps(e.content[:1000], ensure_ascii=False)}}}'
        for e in entries
    )
    return (
        "Для каждой записи ниже напиши краткое саммари (2-3 предложения).\n"
        "Верни ТОЛЬКО JSON-массив без пояснений, формат: "
        '[{"id": <int>, "summary": "..."}, ...].\n\n'
        f"Записи:\n[\n{rows}\n]"
    )


def _build_category_batch_prompt(entries: list[Any], cat_names: str) -> str:
    rows = "\n".join(
        f'  {{"id": {e.id}, "title": {json.dumps(e.title, ensure_ascii=False)}, '
        f'"content": {json.dumps(e.content[:300], ensure_ascii=False)}}}'
        for e in entries
    )
    return (
        "Для каждой записи определи путь категории (например: tools.python). "
        "Можно предлагать новые.\n"
        f"Существующие категории: {cat_names}\n"
        "Верни ТОЛЬКО JSON-массив без пояснений, формат: "
        '[{"id": <int>, "category": "path"}, ...].\n\n'
        f"Записи:\n[\n{rows}\n]"
    )


def _parse_tag_batch(text: str, entries: list[Any]) -> dict[int, list[str]]:
    data = _parse_json_array(text)
    valid_ids = {e.id for e in entries}
    out: dict[int, list[str]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        tags = row.get("tags")
        if not isinstance(rid, int) or rid not in valid_ids:
            continue
        if not isinstance(tags, list):
            continue
        clean = [str(t).strip() for t in tags if str(t).strip()]
        if clean:
            out[rid] = clean
    return out


def _parse_summary_batch(text: str, entries: list[Any]) -> dict[int, str]:
    data = _parse_json_array(text)
    valid_ids = {e.id for e in entries}
    out: dict[int, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        summary = row.get("summary")
        if not isinstance(rid, int) or rid not in valid_ids:
            continue
        if not isinstance(summary, str):
            continue
        if summary.strip():
            out[rid] = summary
    return out


def _parse_category_batch(text: str, entries: list[Any]) -> dict[int, str]:
    data = _parse_json_array(text)
    valid_ids = {e.id for e in entries}
    out: dict[int, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        category = row.get("category")
        if not isinstance(rid, int) or rid not in valid_ids:
            continue
        if not isinstance(category, str):
            continue
        if category.strip():
            out[rid] = category
    return out


def _parse_json_array(text: str) -> list[object]:
    """Parse a JSON array. Raises ``_BatchFailedError`` on any parse error."""
    stripped = text.strip()
    # Tolerate markdown fences the model occasionally produces
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise _BatchFailedError(str(e)) from e
    if not isinstance(data, list):
        raise _BatchFailedError("top-level JSON is not an array")
    return data
