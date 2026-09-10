"""Contract tests for src/llm/providers.py — single source of truth for
LLM providers, model→pricing mapping, and adapter wiring.

These are unit-level tests; they don't construct actual adapters (factory
calls are not invoked here). The registry is the spec — adapters are
mocked elsewhere.
"""

from __future__ import annotations

import pytest
from src.llm.providers import (
    PROVIDERS,
    BillingMode,
    ModelSpec,
    ProviderSpec,
    get_pricing,
    resolve_model,
)


def test_providers_registry_has_known_providers() -> None:
    assert set(PROVIDERS) == {
        "anthropic_api",
        "claude_cli",
        "codex_cli",
        "gemini",
        "huggingface",
        "minimax",
        "nvidia_nim",
        "openrouter",
    }


def test_each_provider_spec_shape_is_valid() -> None:
    for name, spec in PROVIDERS.items():
        assert isinstance(spec, ProviderSpec)
        assert spec.name == name
        assert spec.billing_mode in ("per_token", "subscription", "free_tier")
        assert callable(spec.adapter_factory)
        assert spec.models, f"{name} must declare at least one model"
        for key, model in spec.models.items():
            assert isinstance(model, ModelSpec)
            assert model.api_id
            # The dict key is the canonical api_id of the model
            assert key == model.api_id


def test_billing_mode_literal_values() -> None:
    """BillingMode covers exactly per_token / subscription / free_tier."""
    valid: set[BillingMode] = {"per_token", "subscription", "free_tier"}
    for spec in PROVIDERS.values():
        assert spec.billing_mode in valid


def test_anthropic_api_includes_opus() -> None:
    """Regression — old MODEL_MAP only had haiku/sonnet, breaking strategist
    when routed via Anthropic API. Registry must include Opus."""
    spec = PROVIDERS["anthropic_api"]
    api_ids = set(spec.models)
    assert any("opus" in m for m in api_ids), (
        f"Expected an Opus model in anthropic_api, got: {api_ids}"
    )


def test_anthropic_api_per_token_billing() -> None:
    assert PROVIDERS["anthropic_api"].billing_mode == "per_token"


def test_claude_cli_subscription_billing() -> None:
    assert PROVIDERS["claude_cli"].billing_mode == "subscription"


def test_codex_cli_subscription_billing() -> None:
    assert PROVIDERS["codex_cli"].billing_mode == "subscription"


def test_gemini_free_tier_billing() -> None:
    assert PROVIDERS["gemini"].billing_mode == "free_tier"


def test_anthropic_api_requires_anthropic_key() -> None:
    spec = PROVIDERS["anthropic_api"]
    assert spec.requires_setting == "anthropic_api_key"


def test_gemini_requires_google_key() -> None:
    assert PROVIDERS["gemini"].requires_setting == "google_api_key"


def test_claude_cli_requires_no_setting() -> None:
    """CLI adapter has its own claude_cli_path check in create_router;
    requires_setting=None means the registry doesn't gate on a Settings field."""
    assert PROVIDERS["claude_cli"].requires_setting is None


def test_codex_cli_requires_no_api_setting() -> None:
    """Codex subscription auth lives in Codex CLI state, not an API key setting."""
    assert PROVIDERS["codex_cli"].requires_setting is None


def test_openrouter_per_token_billing() -> None:
    assert PROVIDERS["openrouter"].billing_mode == "per_token"


def test_openrouter_requires_openrouter_key() -> None:
    assert PROVIDERS["openrouter"].requires_setting == "openrouter_api_key"


def test_nvidia_nim_provider_resolves_through_llmrouter() -> None:
    spec = PROVIDERS["nvidia_nim"]
    assert spec.billing_mode == "per_token"
    assert spec.requires_setting == "nvidia_nim_api_key"
    assert resolve_model("nvidia_nim", "nim-default") is not None


def test_huggingface_provider_uses_router_token() -> None:
    spec = PROVIDERS["huggingface"]
    assert spec.billing_mode == "per_token"
    assert spec.requires_setting == "huggingface_api_key"
    assert resolve_model("huggingface", "hf-default") is not None


def test_minimax_provider_uses_openai_compatible_chat() -> None:
    spec = PROVIDERS["minimax"]
    assert spec.billing_mode == "per_token"
    assert spec.requires_setting == "minimax_api_key"
    assert resolve_model("minimax", "m2-7") is not None


def test_openrouter_includes_deepseek_chat() -> None:
    """OpenRouter must ship at least one DeepSeek model out of the box —
    that's the immediate use case for the /compare command."""
    spec = PROVIDERS["openrouter"]
    assert "deepseek/deepseek-chat" in spec.models


def test_openrouter_includes_deepseek_v4_pro() -> None:
    """V4 Pro is the latest flagship DeepSeek (released 2026-04-23/24).
    Registry must expose it so /compare can target it for tests."""
    spec = PROVIDERS["openrouter"]
    assert "deepseek/deepseek-v4-pro" in spec.models


def test_openrouter_includes_deepseek_v4_flash() -> None:
    """V4 Flash is the cheap/fast variant — candidate worker replacement.
    Registry must expose it alongside Pro."""
    spec = PROVIDERS["openrouter"]
    assert "deepseek/deepseek-v4-flash" in spec.models


# --- resolve_model ---


def test_resolve_alias_haiku_to_api_id() -> None:
    spec = resolve_model("anthropic_api", "haiku")
    assert spec is not None
    assert spec.api_id == "claude-haiku-4-5-20251001"


def test_resolve_alias_sonnet_to_api_id() -> None:
    spec = resolve_model("anthropic_api", "sonnet")
    assert spec is not None
    assert spec.api_id == "claude-sonnet-4-6"


def test_resolve_alias_opus_to_api_id() -> None:
    spec = resolve_model("anthropic_api", "opus")
    assert spec is not None
    assert spec.api_id == "claude-opus-4-7"


def test_resolve_api_id_is_identity() -> None:
    spec = resolve_model("anthropic_api", "claude-sonnet-4-6")
    assert spec is not None
    assert spec.api_id == "claude-sonnet-4-6"


def test_resolve_unknown_provider_returns_none() -> None:
    assert resolve_model("openai", "gpt-4") is None


def test_resolve_unknown_model_returns_none() -> None:
    assert resolve_model("anthropic_api", "missing_model_for_tests") is None


def test_resolve_for_cli_passthrough() -> None:
    """CLI accepts short names natively (haiku/sonnet/opus)."""
    spec = resolve_model("claude_cli", "sonnet")
    assert spec is not None
    assert spec.api_id == "sonnet"


def test_resolve_codex_default_model() -> None:
    spec = resolve_model("codex_cli", "default")
    assert spec is not None
    assert spec.api_id == "default"


def test_resolve_codex_gpt_5_5_model() -> None:
    spec = resolve_model("codex_cli", "gpt-5.5")
    assert spec is not None
    assert spec.api_id == "gpt-5.5"


# --- get_pricing ---


def test_get_pricing_haiku_input_080() -> None:
    spec = get_pricing("anthropic_api", "haiku")
    assert spec is not None
    assert spec.input_per_mtok == pytest.approx(0.80)
    assert spec.output_per_mtok == pytest.approx(4.00)


def test_get_pricing_sonnet_input_300() -> None:
    spec = get_pricing("anthropic_api", "sonnet")
    assert spec is not None
    assert spec.input_per_mtok == pytest.approx(3.00)
    assert spec.output_per_mtok == pytest.approx(15.00)


def test_get_pricing_opus_input_1500() -> None:
    spec = get_pricing("anthropic_api", "opus")
    assert spec is not None
    assert spec.input_per_mtok == pytest.approx(15.00)
    assert spec.output_per_mtok == pytest.approx(75.00)


def test_subscription_pricing_is_zero() -> None:
    """CLI subscription does not bill per token — pricing must be zero
    so usage_tracker counts calls but not money."""
    spec = get_pricing("claude_cli", "opus")
    assert spec is not None
    assert spec.input_per_mtok == 0.0
    assert spec.output_per_mtok == 0.0

    codex_spec = get_pricing("codex_cli", "default")
    assert codex_spec is not None
    assert codex_spec.input_per_mtok == 0.0
    assert codex_spec.output_per_mtok == 0.0


def test_free_tier_pricing_is_zero() -> None:
    """Gemini is on the free tier — no per-token cost."""
    spec = get_pricing("gemini", "gemini-2.5-flash-lite")
    assert spec is not None
    assert spec.input_per_mtok == 0.0
    assert spec.output_per_mtok == 0.0


def test_get_pricing_unknown_returns_none() -> None:
    assert get_pricing("anthropic_api", "missing_model_for_tests") is None
    assert get_pricing("openai", "gpt-4") is None


def test_resolve_openrouter_alias_deepseek() -> None:
    """The short alias ``deepseek`` resolves to deepseek-chat — picked so
    ``/compare deepseek <prompt>`` works without typing the vendor prefix."""
    spec = resolve_model("openrouter", "deepseek")
    assert spec is not None
    assert spec.api_id == "deepseek/deepseek-chat"


def test_get_pricing_openrouter_deepseek_chat() -> None:
    spec = get_pricing("openrouter", "deepseek/deepseek-chat")
    assert spec is not None
    assert spec.input_per_mtok > 0
    assert spec.output_per_mtok > 0


def test_resolve_alias_v4_pro() -> None:
    """Short alias ``v4-pro`` resolves to the full DeepSeek V4 Pro api_id."""
    spec = resolve_model("openrouter", "v4-pro")
    assert spec is not None
    assert spec.api_id == "deepseek/deepseek-v4-pro"


def test_resolve_alias_v4_flash() -> None:
    spec = resolve_model("openrouter", "v4-flash")
    assert spec is not None
    assert spec.api_id == "deepseek/deepseek-v4-flash"


def test_get_pricing_v4_pro() -> None:
    """Pricing snapshot from openrouter.ai on 2026-04-25 — refresh if drifts."""
    spec = get_pricing("openrouter", "deepseek/deepseek-v4-pro")
    assert spec is not None
    assert spec.input_per_mtok == pytest.approx(1.74)
    assert spec.output_per_mtok == pytest.approx(3.48)


def test_get_pricing_v4_flash() -> None:
    spec = get_pricing("openrouter", "deepseek/deepseek-v4-flash")
    assert spec is not None
    assert spec.input_per_mtok == pytest.approx(0.14)
    assert spec.output_per_mtok == pytest.approx(0.28)


# --- ModelSpec defaults ---


def test_model_spec_defaults() -> None:
    """Default ModelSpec has zero pricing, identity emoji, no aliases."""
    m = ModelSpec(api_id="x")
    assert m.api_id == "x"
    assert m.input_per_mtok == 0.0
    assert m.output_per_mtok == 0.0
    assert m.cache_write_mult == pytest.approx(1.25)
    assert m.cache_read_mult == pytest.approx(0.1)
    assert m.aliases == ()


def test_model_spec_is_frozen() -> None:
    """Specs are immutable so they can be shared safely across the app."""
    m = ModelSpec(api_id="x")
    with pytest.raises((AttributeError, TypeError)):
        m.api_id = "y"  # type: ignore[misc]


def test_provider_spec_is_frozen() -> None:
    spec = PROVIDERS["anthropic_api"]
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "renamed"  # type: ignore[misc]
