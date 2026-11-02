"""Market Pulse selection: which questions a polling pass forecasts or re-forecasts, and group labels."""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("METACULUS_TOKEN", "offline-test")
import gigbot_bot as g  # noqa: E402
from forecasting_tools import NumericQuestion  # noqa: E402

NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


def q(forecast_at=None, text="What will the S&P 500 close at on Nov 20, 2026?", label=None):
    api = {"question": {"my_forecasts": {"latest": {"start_time": forecast_at.timestamp()}}}} if forecast_at else {}
    return NumericQuestion(
        question_text=text, id_of_question=1, id_of_post=1, already_forecasted=forecast_at is not None,
        group_question_option=label, api_json=api, upper_bound=10, lower_bound=0,
        open_upper_bound=True, open_lower_bound=True,
    )


def test_new_question_is_always_forecast():
    assert g.needs_forecast(q(), None, NOW)
    assert g.needs_forecast(q(), 48, NOW)


def test_forecast_once_tournaments_never_reforecast():
    assert not g.needs_forecast(q(NOW - timedelta(days=30)), None, NOW)


def test_market_pulse_reforecasts_only_stale_ones():
    assert not g.needs_forecast(q(NOW - timedelta(hours=47)), 48, NOW)
    assert g.needs_forecast(q(NOW - timedelta(hours=49)), 48, NOW)


def test_group_label_is_added_once():
    sub = g.label_subquestion(q(text="Closing value on Nov 20, 2026", label="Nasdaq-100"))
    assert sub.question_text.endswith("[Nasdaq-100]")
    again = g.label_subquestion(sub)
    assert again.question_text.count("Nasdaq-100") == 1
    assert g.label_subquestion(q(label=None)).question_text.endswith("2026?")


def test_market_pulse_is_polled_last():
    targets = g.poll_targets()
    assert targets[0] == (g.SEASONAL_TOURNAMENT, None)
    assert targets[1] == (g.MINIBENCH_TOURNAMENT, None)
    if g.MARKET_PULSE_ON:
        assert targets[2] == (g.MARKET_PULSE_TOURNAMENT, g.REFORECAST_HOURS)
