"""Tests for ``LLMRouter.generate_oneoff`` — explicit provider+model
dispatch used by the /compare command.

Unlike ``generate(LLMRequest(tier=...))``, ``generate_oneoff`` does not
go through tier mapping. It builds (or reuses) the adapter from the
``PROVIDERS`` registry and dispatches directly. This lets a caller hit
any provider/model from anywhere in the app without reshaping its tier
config.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.llm.protocols import LLMError, LLMResponse, LLMUsage
from src.llm.router import LLMRouter


def _empty_router(**kwargs: object) -> LLMRouter:
    """Build a router whose tier adapters are never used in these tests."""
    adapter = MagicMock()
    adapter.name = "stub"
    adapter.default_model = "stub"
    adapter.generate = AsyncMock()
    return LLMRouter(
        adapters={"worker": adapter, "analyst": adapter, "strategist": adapter},
        models={"worker": "stub", "analyst": "stub", "strategist": "stub"},
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch ``PROVIDERS`` with a single fake provider whose adapter
    factory returns a controllable ``AsyncMock``. Frees the test from
    the real OpenRouter / Anthropic registry shape."""
    from src.llm import providers as p
    from src.llm import router

    fake_adapter = MagicMock()
    fake_adapter.name = "fake"
    fake_adapter.generate = AsyncMock(
        return_value=LLMResponse(
            text="ok", model="x/y", usage=LLMUsage(input_tokens=3, output_tokens=4)
        )
    )

    factory = MagicMock(return_value=fake_adapter)
    fake_spec = p.ProviderSpec(
        name="fake",
        billing_mode="per_token",
        models={
            "x/y": p.ModelSpec(
                api_id="x/y",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
                aliases=("y-alias",),
            ),
        },
        adapter_factory=factory,
        requires_setting=None,
    )
    monkeypatch.setitem(p.PROVIDERS, "fake", fake_spec)
    monkeypatch.setattr(
        router,
        "PROVIDERS",
        p.PROVIDERS,
    )
    return fake_adapter


class TestGenerateOneoff:
    async def test_dispatches_to_adapter_from_registry(
        self, stub_provider: AsyncMock
    ) -> None:
        router = _empty_router()
        resp = await router.generate_oneoff(
            provider="fake", model="x/y", prompt="hi", caller="test"
        )

        assert resp.text == "ok"
        stub_provider.generate.assert_awaited_once()
        sent = stub_provider.generate.call_args.args[0]
        assert sent.prompt == "hi"
        assert sent.model == "x/y"
        assert sent.caller == "test"

    async def test_resolves_alias_before_dispatch(
        self, stub_provider: AsyncMock
    ) -> None:
        router = _empty_router()
        await router.generate_oneoff(
            provider="fake", model="y-alias", prompt="hi", caller="test"
        )

        sent = stub_provider.generate.call_args.args[0]
        # Alias "y-alias" → api_id "x/y"
        assert sent.model == "x/y"

    async def test_unknown_provider_raises(self) -> None:
        router = _empty_router()
        with pytest.raises(LLMError, match=r"(?i)provider"):
            await router.generate_oneoff(
                provider="not-a-provider", model="x", prompt="hi"
            )

    async def test_passes_system_prompt(self, stub_provider: AsyncMock) -> None:
        router = _empty_router()
        await router.generate_oneoff(
            provider="fake",
            model="x/y",
            prompt="hi",
            system="be terse",
            caller="t",
        )

        sent = stub_provider.generate.call_args.args[0]
        assert sent.system == "be terse"

    async def test_records_usage_per_token(self, stub_provider: AsyncMock) -> None:
        tracker = MagicMock()
        router = _empty_router(usage_tracker=tracker)

        await router.generate_oneoff(
            provider="fake", model="x/y", prompt="hi", caller="compare/main"
        )

        tracker.record_api_call.assert_called_once()
        kwargs = tracker.record_api_call.call_args.kwargs
        assert kwargs["provider"] == "fake"
        assert kwargs["caller"] == "compare/main"
        assert kwargs["input_tokens"] == 3
        assert kwargs["output_tokens"] == 4

    async def test_caches_adapter_across_calls(
        self, stub_provider: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm import providers as p

        factory = p.PROVIDERS["fake"].adapter_factory
        router = _empty_router()

        await router.generate_oneoff(provider="fake", model="x/y", prompt="a")
        await router.generate_oneoff(provider="fake", model="x/y", prompt="b")

        # The factory must run once even though we hit the provider twice
        assert factory.call_count == 1  # type: ignore[attr-defined]
