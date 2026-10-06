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
