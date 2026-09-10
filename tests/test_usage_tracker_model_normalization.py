"""Bucket normalization: alias and api_id collapse into one bucket via
the registry. Buckets are ``"<provider>/<api_id>"``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.monitoring.usage_tracker import UsageTracker

if TYPE_CHECKING:
    from pathlib import Path

HAIKU = "anthropic_api/claude-haiku-4-5-20251001"
SONNET = "anthropic_api/claude-sonnet-4-6"
OPUS = "anthropic_api/claude-opus-4-7"


def test_full_model_id_computes_nonzero_cost(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )

    today = tracker.get_today()
    assert today.cost_usd > 4.0


def test_full_haiku_model_id_computes_nonzero_cost(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )

    today = tracker.get_today()
    # Haiku $0.80/M input + $4.00/M output @ 100k = $0.80 + $0.40 = $1.20
    assert today.cost_usd > 1.0


def test_full_opus_model_id_computes_nonzero_cost(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )

    today = tracker.get_today()
    # Opus $15/M input + $75/M output @ 100k = $15 + $7.5 = $22.50
    assert today.cost_usd > 22.0


def test_full_and_short_model_ids_aggregate_to_same_bucket(tmp_path: Path) -> None:
    """Calls under alias ``sonnet`` and api_id ``claude-sonnet-4-6``
    collapse to the same bucket — pricing aggregates correctly."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call(
        "claude-sonnet-4-6", input_tokens=500_000, output_tokens=50_000
    )
    tracker.record_api_call("sonnet", input_tokens=500_000, output_tokens=50_000)

    today = tracker.get_today()
    assert today.api_calls[SONNET] == 2
    assert today.input_tokens[SONNET] == 1_000_000


def test_load_migrates_old_full_model_names(tmp_path: Path) -> None:
    """A monthly file written before the provider-aware refactor stored
    full api_ids without a provider prefix. Loader migrates each bucket
    to ``"anthropic_api/<api_id>"`` so cost_usd recomputes correctly."""
    data_dir = tmp_path / "monitoring"
    data_dir.mkdir(parents=True)
    month = datetime.now(tz=UTC).strftime("%Y-%m")
    today_key = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    old_data = {
        today_key: {
            "api_calls": {"claude-sonnet-4-6": 3, "claude-haiku-4-5-20251001": 5},
            "cli_sessions": 0,
            "cli_calls": 0,
            "gemini_calls": 0,
            "input_tokens": {
                "claude-sonnet-4-6": 1_000_000,
                "claude-haiku-4-5-20251001": 500_000,
            },
            "output_tokens": {
                "claude-sonnet-4-6": 100_000,
                "claude-haiku-4-5-20251001": 50_000,
            },
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "caller_counts": {},
        }
    }
    (data_dir / f"usage_{month}.json").write_text(
        json.dumps(old_data), encoding="utf-8"
    )

    tracker = UsageTracker(data_dir)
    today = tracker.get_today()

    assert today.api_calls.get(SONNET) == 3
    assert today.api_calls.get(HAIKU) == 5
    assert today.input_tokens.get(SONNET) == 1_000_000
    assert today.input_tokens.get(HAIKU) == 500_000
    assert today.cost_usd > 4.0


def test_legacy_alias_buckets_load_into_anthropic_api(tmp_path: Path) -> None:
    """Old buckets ``"haiku"`` / ``"sonnet"`` / ``"opus"`` migrate into
    the canonical ``anthropic_api`` provider with the api_id."""
    data_dir = tmp_path / "monitoring"
    data_dir.mkdir(parents=True)
    month = datetime.now(tz=UTC).strftime("%Y-%m")
    today_key = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    old_data = {
        today_key: {
            "api_calls": {"haiku": 5, "sonnet": 3, "opus": 1},
            "input_tokens": {"haiku": 100_000, "sonnet": 50_000, "opus": 10_000},
            "output_tokens": {"haiku": 50_000, "sonnet": 20_000, "opus": 5_000},
            "cli_sessions": 0,
            "cli_calls": 0,
            "gemini_calls": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "caller_counts": {},
        }
    }
    (data_dir / f"usage_{month}.json").write_text(
        json.dumps(old_data), encoding="utf-8"
    )

    tracker = UsageTracker(data_dir)
    today = tracker.get_today()

    assert today.api_calls.get(HAIKU) == 5
    assert today.api_calls.get(SONNET) == 3
    assert today.api_calls.get(OPUS) == 1


def test_unknown_model_name_preserved_as_is(tmp_path: Path) -> None:
    """An unknown model is preserved with zero cost; known calls still bill."""
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_api_call("mystery-model", input_tokens=1000, output_tokens=500)
    tracker.record_api_call(
        "claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=100_000
    )

    today = tracker.get_today()
    assert "anthropic_api/mystery-model" in today.api_calls
    assert SONNET in today.api_calls
    assert today.cost_usd > 4.0
