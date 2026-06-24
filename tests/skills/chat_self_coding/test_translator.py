"""Tests for ``chat_self_coding.translator`` (Phase 40).

The translator turns technical audit-log text (commit summaries,
diff bullet points, error messages) into orchestrator-language
descriptions for the Telegram block messages. LLM worker tier
(Haiku) does the actual translation; we test the contract — call
shape, caching, kind propagation — not the language itself, since
LLM output is not testable without a live model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


def _mock_llm(reply: str = "Расширила систему пресетов.") -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_llm_resp(reply))
    return llm


class FakeRedis:
    """Minimal in-memory ``redis.asyncio`` stand-in (get / set with ex)."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def get(self, key: str) -> str | None:
        self.calls.append(("get", (key,)))
        return self._kv.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.calls.append(("set", (key, value, ex)))
        self._kv[key] = value


class FakeRedisBytes(FakeRedis):
    async def get(self, key: str) -> bytes | None:  # type: ignore[override]
        self.calls.append(("get", (key,)))
        v = self._kv.get(key)
        return v.encode("utf-8") if v is not None else None


# ---------------------------------------------------------------------------
# TranslationKind enum
# ---------------------------------------------------------------------------


class TestTranslationKind:
    def test_three_canonical_kinds(self) -> None:
        from src.skills.chat_self_coding.translator import TranslationKind

        assert {k.value for k in TranslationKind} == {
            "spec_summary",
            "commit_diff",
            "error_message",
        }


# ---------------------------------------------------------------------------
# LLMTranslator contract
# ---------------------------------------------------------------------------


class TestLLMTranslatorContract:
    async def test_translate_calls_worker_tier_zero_temperature(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("Архитектурное описание.")
        translator = LLMTranslator(llm_router=llm)
        await translator.translate(
            "Added budget_seconds field.",
            kind=TranslationKind.COMMIT_DIFF,
        )
        request = llm.generate.call_args.args[0]
        assert request.tier == "worker"
        assert request.temperature == 0.0
        assert request.caller == "chat_self_coding_translator"

    async def test_translate_strips_whitespace(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("\n  Расширила систему пресетов.\n  ")
        translator = LLMTranslator(llm_router=llm)
        result = await translator.translate("x", kind=TranslationKind.SPEC_SUMMARY)
        assert result == "Расширила систему пресетов."

    async def test_translate_includes_kind_in_prompt(self) -> None:
        """Different kinds need different prompting — the kind tag must
        reach the LLM somehow (system or user prompt)."""
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("ok")
        translator = LLMTranslator(llm_router=llm)
        await translator.translate("text", kind=TranslationKind.ERROR_MESSAGE)
        request = llm.generate.call_args.args[0]
        combined = (request.system or "") + " " + request.prompt
        assert (
            "error_message" in combined.lower()
            or "ошибк" in combined.lower()
            or "неуда" in combined.lower()
        )

    async def test_translate_includes_technical_text_in_prompt(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("translated")
        translator = LLMTranslator(llm_router=llm)
        await translator.translate(
            "Added budget_seconds field to ResearchPreset.",
            kind=TranslationKind.COMMIT_DIFF,
        )
        request = llm.generate.call_args.args[0]
        assert "budget_seconds" in request.prompt

    async def test_banned_terms_include_markdown_html_regex_and_parse_mode(
        self,
    ) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("ok")
        translator = LLMTranslator(llm_router=llm)
        await translator.translate(
            "markdown parse_mode regex", kind=TranslationKind.ERROR_MESSAGE
        )
        request = llm.generate.call_args.args[0]
        banned = request.system.lower()
        for term in ("markdown", "html", "regex", "parse_mode"):
            assert term in banned

    async def test_translate_returns_original_when_llm_returns_empty(self) -> None:
        """LLM degenerate case: if it emits whitespace-only, fall back to
        the original technical text rather than show an empty bubble."""
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("   \n   ")
        translator = LLMTranslator(llm_router=llm)
        result = await translator.translate(
            "Added budget_seconds field.",
            kind=TranslationKind.COMMIT_DIFF,
        )
        assert result == "Added budget_seconds field."


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestTranslatorCaching:
    async def test_repeated_input_hits_cache_no_second_llm_call(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("Кэшируемое описание.")
        redis = FakeRedis()
        translator = LLMTranslator(llm_router=llm, redis=redis)

        first = await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        second = await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        assert first == second
        # LLM consulted exactly once — second call hit the cache.
        llm.generate.assert_awaited_once()

    async def test_no_redis_means_every_call_hits_llm(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("ok")
        translator = LLMTranslator(llm_router=llm, redis=None)

        await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        assert llm.generate.await_count == 2

    async def test_cache_key_distinguishes_kind(self) -> None:
        """Same technical text under a different kind yields a different
        translation prompt — so caches must not collide."""
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("any")
        redis = FakeRedis()
        translator = LLMTranslator(llm_router=llm, redis=redis)

        await translator.translate("same", kind=TranslationKind.SPEC_SUMMARY)
        await translator.translate("same", kind=TranslationKind.COMMIT_DIFF)
        assert llm.generate.await_count == 2

    async def test_cache_handles_bytes_payload(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("ok")
        redis = FakeRedisBytes()
        translator = LLMTranslator(llm_router=llm, redis=redis)
        await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        # Second call must read bytes back and return the same string.
        result = await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        assert result == "ok"
        llm.generate.assert_awaited_once()

    async def test_redis_failure_falls_back_to_llm(self) -> None:
        """If Redis raises on get/set, translation should still succeed —
        caching is best-effort, not load-bearing."""
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
        )

        class BrokenRedis:
            async def get(self, key: str) -> str | None:
                raise ConnectionError("redis down")

            async def set(self, key: str, value: str, ex: int | None = None) -> None:
                raise ConnectionError("redis down")

        llm = _mock_llm("Описание.")
        translator = LLMTranslator(llm_router=llm, redis=BrokenRedis())
        result = await translator.translate("tech", kind=TranslationKind.COMMIT_DIFF)
        assert result == "Описание."

    async def test_cache_set_uses_default_ttl(self) -> None:
        from src.skills.chat_self_coding.translator import (
            DEFAULT_CACHE_TTL_SECONDS,
            LLMTranslator,
            TranslationKind,
        )

        llm = _mock_llm("ok")
        redis = FakeRedis()
        translator = LLMTranslator(llm_router=llm, redis=redis)
        await translator.translate("x", kind=TranslationKind.COMMIT_DIFF)
        set_call = next(c for c in redis.calls if c[0] == "set")
        assert set_call[1][2] == DEFAULT_CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


class TestProtocol:
    async def test_translator_satisfies_protocol(self) -> None:
        from src.skills.chat_self_coding.translator import (
            LLMTranslator,
            TranslationKind,
            Translator,
        )

        llm = _mock_llm("ok")
        translator: Translator = LLMTranslator(llm_router=llm)
        result = await translator.translate("x", kind=TranslationKind.SPEC_SUMMARY)
        assert isinstance(result, str)
