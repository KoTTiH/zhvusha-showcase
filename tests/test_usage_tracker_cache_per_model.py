"""Per-model cache cost tracking. Buckets are ``"<provider>/<api_id>"``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.monitoring.usage_tracker import DayStats, UsageTracker

if TYPE_CHECKING:
    from pathlib import Path

HAIKU = "anthropic_api/claude-haiku-4-5-20251001"
SONNET = "anthropic_api/claude-sonnet-4-6"


def test_cache_tokens_bucketed_per_model(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku", cache_read_tokens=1000, cache_write_tokens=200)
    tracker.record_api_call("sonnet", cache_read_tokens=500, cache_write_tokens=100)

    today = tracker.get_today()
    assert today.cache_read_tokens_by_model == {HAIKU: 1000, SONNET: 500}
    assert today.cache_write_tokens_by_model == {HAIKU: 200, SONNET: 100}


def test_cache_saved_usd_uses_model_specific_prices(tmp_path: Path) -> None:
    """Savings for a haiku-only cache must be priced at haiku input rate
    ($0.80/M), not sonnet ($3.00/M)."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku", cache_read_tokens=1_000_000)

    today = tracker.get_today()
    # Haiku input $0.80/M; saved = 90% of full price = $0.72
    assert abs(today.cache_saved_usd - 0.72) < 0.001


def test_cache_write_uses_5min_multiplier_125(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", cache_write_tokens=1_000_000)

    today = tracker.get_today()
    # Sonnet input $3.00/M x 1.25 = $3.75 for 1M write tokens
    assert abs(today.cost_usd - 3.75) < 0.01


def test_cost_sums_haiku_and_sonnet_cache_separately(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku", cache_read_tokens=1_000_000)
    tracker.record_api_call("sonnet", cache_read_tokens=1_000_000)

    today = tracker.get_today()
    # haiku 1M x $0.80 x 0.1 = $0.08
    # sonnet 1M x $3.00 x 0.1 = $0.30
    assert abs(today.cost_usd - 0.38) < 0.01


def test_legacy_scalar_cache_fields_still_load(tmp_path: Path) -> None:
    """Older monthly files stored ``cache_read_tokens`` as a single int
    and a bucket without provider prefix. Loader migrates them to the
    new ``provider/api_id`` schema, attributing cache to sonnet."""
    data_dir = tmp_path / "monitoring"
    data_dir.mkdir(parents=True)
    month = datetime.now(tz=UTC).strftime("%Y-%m")
    today_key = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    old_data = {
        today_key: {
            "api_calls": {"sonnet": 1},
            "input_tokens": {"sonnet": 100},
            "output_tokens": {"sonnet": 50},
            "cache_read_tokens": 10_000,
            "cache_write_tokens": 2_000,
            "caller_counts": {},
            "cli_sessions": 0,
            "cli_calls": 0,
            "gemini_calls": 0,
        }
    }
    (data_dir / f"usage_{month}.json").write_text(
        json.dumps(old_data), encoding="utf-8"
    )

    tracker = UsageTracker(data_dir)
    today = tracker.get_today()
    assert today.cache_read_tokens_by_model == {SONNET: 10_000}
    assert today.cache_write_tokens_by_model == {SONNET: 2_000}


def test_cache_hit_rate_still_reported(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("sonnet", cache_read_tokens=900, cache_write_tokens=100)
    tracker.record_api_call("haiku", cache_read_tokens=180, cache_write_tokens=20)

    today = tracker.get_today()
    # read 1080 / (1080 + 120) = 0.9
    assert abs(today.cache_hit_rate - 0.9) < 0.001


def test_to_dict_writes_per_model_cache(tmp_path: Path) -> None:
    """Serialised output round-trips through DayStats.from_dict."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("haiku", cache_read_tokens=500, cache_write_tokens=100)

    today = tracker.get_today()
    raw = today.to_dict()
    stats2 = DayStats.from_dict(raw)
    assert stats2.cache_read_tokens_by_model == {HAIKU: 500}
    assert stats2.cache_write_tokens_by_model == {HAIKU: 100}
