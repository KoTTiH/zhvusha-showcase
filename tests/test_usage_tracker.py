"""Tests for UsageTracker. Buckets are keyed ``"<provider>/<api_id>"``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.monitoring.usage_tracker import UsageTracker

if TYPE_CHECKING:
    from pathlib import Path

HAIKU = "anthropic_api/claude-haiku-4-5-20251001"
SONNET = "anthropic_api/claude-sonnet-4-6"
OPUS = "anthropic_api/claude-opus-4-7"


def test_record_api_call_increments_count(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku")
    tracker.record_api_call("haiku")
    tracker.record_api_call("sonnet")

    today = tracker.get_today()
    assert today.api_calls[HAIKU] == 2
    assert today.api_calls[SONNET] == 1


def test_record_api_call_with_tokens(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", input_tokens=500, output_tokens=200)
    tracker.record_api_call("sonnet", input_tokens=300, output_tokens=100)

    today = tracker.get_today()
    assert today.input_tokens[SONNET] == 800
    assert today.output_tokens[SONNET] == 300
    assert today.api_calls[SONNET] == 2


def test_record_api_call_with_cache_tokens(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "sonnet",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=1500,
        cache_write_tokens=0,
    )
    tracker.record_api_call(
        "sonnet",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=1500,
        cache_write_tokens=0,
    )

    today = tracker.get_today()
    assert today.cache_read_tokens == 3000
    assert today.cache_write_tokens == 0


def test_cost_usd_from_tokens(tmp_path: Path) -> None:
    """Cost is calculated from tokens via the provider registry."""
    tracker = UsageTracker(tmp_path / "monitoring")
    # 1M input tokens of sonnet = $3.00, 100k output = $1.50
    tracker.record_api_call("sonnet", input_tokens=1_000_000, output_tokens=100_000)

    today = tracker.get_today()
    assert abs(today.cost_usd - 4.50) < 0.01


def test_cost_usd_zero_without_tokens(tmp_path: Path) -> None:
    """Without tokens (count-only call) cost stays zero — there is no
    per-call legacy fallback any more."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku")
    tracker.record_api_call("sonnet")

    assert tracker.get_today().cost_usd == 0.0


def test_cache_hit_rate(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", cache_read_tokens=900, cache_write_tokens=100)

    today = tracker.get_today()
    assert abs(today.cache_hit_rate - 0.9) < 0.001


def test_cache_hit_rate_zero_when_no_cache(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", input_tokens=100, output_tokens=50)

    assert tracker.get_today().cache_hit_rate == 0.0


def test_cache_saved_usd(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    # 10k cache read tokens at sonnet input $3.00/M
    # Saved: 90% of $0.03 = $0.027
    tracker.record_api_call("sonnet", cache_read_tokens=10000)

    today = tracker.get_today()
    assert abs(today.cache_saved_usd - 0.027) < 0.001


def test_get_today_empty(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    today = tracker.get_today()

    assert today.total_api == 0
    assert today.cli_sessions == 0
    assert today.gemini_calls == 0
    assert today.cost_usd == 0.0
    assert today.cache_hit_rate == 0.0


def test_record_cli_session(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_cli_session()

    assert tracker.get_today().cli_sessions == 1


def test_record_gemini_call(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_gemini_call()
    tracker.record_gemini_call()

    assert tracker.get_today().gemini_calls == 2


def test_get_month_total(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", input_tokens=1_000_000, output_tokens=100_000)

    assert tracker.get_month_total() > 0


def test_persists_to_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "monitoring"
    tracker = UsageTracker(data_dir)
    tracker.record_api_call(
        "sonnet", input_tokens=500, output_tokens=200, cache_read_tokens=1000
    )
    tracker.record_cli_session()

    tracker2 = UsageTracker(data_dir)
    today = tracker2.get_today()

    assert today.api_calls.get(SONNET) == 1
    assert today.input_tokens.get(SONNET) == 500
    assert today.cache_read_tokens == 1000
    assert today.cli_sessions == 1


def test_backward_compat_old_format(tmp_path: Path) -> None:
    """Old JSON files (no provider prefix on bucket keys) load and migrate
    to the canonical ``"anthropic_api/<api_id>"`` form."""
    data_dir = tmp_path / "monitoring"
    data_dir.mkdir(parents=True)

    from datetime import UTC, datetime

    month = datetime.now(tz=UTC).strftime("%Y-%m")
    today_key = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    old_data = {
        today_key: {
            "api_calls": {"haiku": 5, "sonnet": 3},
            "cli_sessions": 1,
            "gemini_calls": 2,
        }
    }
    (data_dir / f"usage_{month}.json").write_text(
        json.dumps(old_data), encoding="utf-8"
    )

    tracker = UsageTracker(data_dir)
    today = tracker.get_today()

    assert today.api_calls[HAIKU] == 5
    assert today.api_calls[SONNET] == 3
    assert today.input_tokens == {}
    assert today.cache_read_tokens == 0
    # No more per-call legacy fallback — count-only files cost zero
    assert today.cost_usd == 0.0


def test_total_api_property(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku")
    tracker.record_api_call("haiku")
    tracker.record_api_call("sonnet")

    assert tracker.get_today().total_api == 3


def test_get_calls_in_last_hour(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    assert tracker.get_calls_in_last_hour() == 0

    tracker.record_api_call("haiku")
    tracker.record_api_call("sonnet")
    tracker.record_cli_call()
    tracker.record_gemini_call()

    assert tracker.get_calls_in_last_hour() == 4


def test_get_calls_in_last_hour_prunes_old(tmp_path: Path) -> None:
    import time

    tracker = UsageTracker(tmp_path / "monitoring")
    tracker._call_timestamps.append(time.monotonic() - 3700)
    tracker.record_api_call("haiku")

    assert tracker.get_calls_in_last_hour() == 1


def test_record_api_call_with_explicit_provider(tmp_path: Path) -> None:
    """The router supplies provider explicitly — bucket key is still the
    canonical ``provider/api_id`` form."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "claude-opus-4-7",
        provider="anthropic_api",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )

    today = tracker.get_today()
    assert today.api_calls[OPUS] == 1
    # Opus input $15/M + output $75/M @ 100k = $15.00 + $7.50 = $22.50
    assert abs(today.cost_usd - 22.50) < 0.01


def test_alias_and_api_id_aggregate_to_same_bucket(tmp_path: Path) -> None:
    """Calls under ``haiku`` (alias) and ``claude-haiku-4-5-20251001``
    (api_id) collapse into one bucket via the registry."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku", input_tokens=500_000)
    tracker.record_api_call("claude-haiku-4-5-20251001", input_tokens=500_000)

    today = tracker.get_today()
    assert today.api_calls[HAIKU] == 2
    assert today.input_tokens[HAIKU] == 1_000_000


def test_unknown_model_lands_in_passthrough_bucket(tmp_path: Path) -> None:
    """Unknown model is preserved with zero cost — known calls still bill."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("mystery-model", input_tokens=1000)
    tracker.record_api_call("sonnet", input_tokens=1_000_000, output_tokens=100_000)

    today = tracker.get_today()
    assert today.api_calls.get("anthropic_api/mystery-model") == 1
    assert today.api_calls.get(SONNET) == 1
    # mystery contributes zero, sonnet $4.50
    assert abs(today.cost_usd - 4.50) < 0.01
