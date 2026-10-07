"""Unit tests for gemini_budget (no network). Run: .venv/bin/python -m pytest tests/test_gemini_budget.py -q"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gemini_budget as gb  # noqa: E402


def test_pacific_day_rolls_at_midnight_pacific():
    # 06:59Z on Oct 8 is 23:59 Pacific on Oct 7 (PDT); 07:00Z is Oct 8.
    assert gb.pacific_today(datetime(2026, 10, 8, 6, 59, tzinfo=timezone.utc)) == "2026-10-07"
    assert gb.pacific_today(datetime(2026, 10, 8, 7, 0, tzinfo=timezone.utc)) == "2026-10-08"


def test_parse_usage_same_day_other_day_and_garbage():
    assert gb.parse_usage('{"day": "2026-10-07", "calls": 42}', "2026-10-07") == 42
    assert gb.parse_usage('{"day": "2026-10-06", "calls": 42}', "2026-10-07") == 0
    assert gb.parse_usage("", "2026-10-07") == 0
    assert gb.parse_usage(None, "2026-10-07") == 0
    assert gb.parse_usage("not json", "2026-10-07") == 0
    assert gb.parse_usage('{"day": "2026-10-07", "calls": -5}', "2026-10-07") == 0


def test_format_round_trips():
    s = gb.format_usage("2026-10-07", 7)
    assert gb.parse_usage(s, "2026-10-07") == 7


def test_budget_takes_until_cap_then_raises_and_stays_exhausted():
    b = gb.Budget(cap=5, used_at_start=3, today="2026-10-07")
    assert b.remaining == 2
    b.take(); b.take()
    assert b.remaining == 0 and b.used_this_run == 2
    with pytest.raises(gb.GeminiBudgetExhausted):
        b.take()
    with pytest.raises(gb.GeminiBudgetExhausted):
        b.take()
    assert b.used_this_run == 2  # failed takes don't count
    assert "5/5" in b.summary()


def test_budget_exhaust_on_daily_429_blocks_further_calls():
    b = gb.Budget(cap=300, used_at_start=0, today="2026-10-07")
    b.take()
    b.exhaust("daily quota 429")
    with pytest.raises(gb.GeminiBudgetExhausted):
        b.take()
    assert "daily quota 429" in b.summary()


def test_daily_quota_detection():
    assert gb.is_daily_quota_error("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert gb.is_daily_quota_error("You exceeded your daily limit")
    assert not gb.is_daily_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    assert not gb.is_daily_quota_error("You exceeded your current quota")


def test_budget_from_env_uses_cap_and_stored_usage(monkeypatch):
    today = gb.pacific_today()
    env = {"GEMINI_DAILY_CAP": "10", "GEMINI_USAGE": gb.format_usage(today, 4)}
    b = gb.budget_from_env(env)
    assert b.cap == 10 and b.used_at_start == 4 and b.remaining == 6
    b2 = gb.budget_from_env({"GEMINI_USAGE": ""})
    assert b2.cap == gb.DEFAULT_DAILY_CAP and b2.used_at_start == 0


def test_record_usage_without_token_is_a_noop():
    assert gb.record_usage(3, env={}) is None
    assert gb.record_usage(0, env={"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "a/b"}) is None
