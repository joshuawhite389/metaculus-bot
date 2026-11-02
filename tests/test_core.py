"""Unit tests for forecaster_core (no network, no LLM).  Run: .venv/bin/python -m pytest tests/test_core.py -q"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import forecaster_core as core  # noqa: E402


def test_parse_takes_last_probability():
    text = "Base rate maybe Probability: 20% ... after thinking.\n**Probability: 35%**"
    assert core.parse_probability(text) == 0.35
    assert core.parse_probability("Probability: 7.5 %") == 0.075
    assert core.parse_probability("no number here") is None
    assert core.parse_probability("Probability: 140%") is None


def test_aggregations():
    assert core.aggregate_median([0.1, 0.2, 0.9]) == 0.2
    # mean of log-odds of symmetric pair is 0.5
    assert abs(core.aggregate_mean_logodds([0.2, 0.8]) - 0.5) < 1e-9
    # log-odds mean is pulled toward the confident sample more than the plain mean
    assert core.aggregate_mean_logodds([0.5, 0.99]) > 0.5


def test_final_binary_is_clamped():
    assert core.P_MIN <= core.aggregate_binary([0.0001] * 5) <= core.P_MAX
    assert core.P_MIN <= core.aggregate_binary([0.9999] * 5) <= core.P_MAX


def test_fit_calibration_recovers_known_map():
    rng = random.Random(0)
    true_a, true_b = 1.6, -0.4
    preds, ys = [], []
    for _ in range(4000):
        p = rng.uniform(0.03, 0.97)
        q = core.calibrate(p, true_a, true_b)
        preds.append(p)
        ys.append(1 if rng.random() < q else 0)
    a, b = core.fit_calibration(preds, ys)
    assert abs(a - true_a) < 0.15 and abs(b - true_b) < 0.15


def test_scores():
    assert abs(core.brier(0.7, 1) - 0.09) < 1e-12
    assert core.log_score(0.5, 0) < 0
    assert core.log_score(0.9, 1) > core.log_score(0.6, 1)


def test_parse_percentiles_takes_the_last_clean_block():
    text = """AMD: Percentile 10: 0.5
Percentile 20: 0.6
Percentile 40: 0.7
Percentile 60: 0.8
Percentile 80: 0.9
Percentile 90: 1.0
Final answer:
Percentile 10: -0.15
Percentile 20: 0.02 $
Percentile 40: 1,200.5
Percentile 60: 1,300 (median-ish)
**Percentile 80:** 1400
Percentile 90: 1500 USD
"""
    assert core.parse_percentiles(text) == [(0.1, -0.15), (0.2, 0.02), (0.4, 1200.5), (0.6, 1300.0), (0.8, 1400.0), (0.9, 1500.0)]


def test_parse_percentiles_refuses_unit_words_and_bad_order():
    base = "Percentile 10: 1\nPercentile 20: 2\nPercentile 40: 3\nPercentile 60: 4\nPercentile 80: 5\nPercentile 90: {}\n"
    assert core.parse_percentiles(base.format("6 billion")) is None
    assert core.parse_percentiles(base.format("6 bps"), unit="bps") is not None
    assert core.parse_percentiles(base.format("3")) is None  # not increasing
    assert core.parse_percentiles("Percentile 10: 1\nPercentile 20: 2") is None
    assert core.parse_percentiles("no numbers here") is None


def test_plausible_scale_catches_unit_mismatch():
    pcts = [(0.1, 89e9), (0.2, 90e9), (0.4, 91e9), (0.6, 92e9), (0.8, 93e9), (0.9, 94e9)]
    assert not core.plausible_scale(pcts, 60, 120)  # question asked in billions
    assert core.plausible_scale(pcts, 60e9, 120e9)
    assert core.plausible_scale([(0.1, -9), (0.2, -5), (0.4, -1), (0.6, 1), (0.8, 5), (0.9, 9)], -20, 20)
