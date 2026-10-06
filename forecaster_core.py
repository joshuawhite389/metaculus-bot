"""
Framework-free forecasting logic shared by the live bot (gigbot_bot.py) and the backtest
(../backtest/run_backtest.py), so the backtest measures the same prompts and maths the bot runs.

What differs from the Metaculus template (binary questions):
- Prompt: outside view first (reference class + base rate), then inside view, then an explicit
  check of how much time is left for the event to happen.
- Aggregation: mean of log-odds across samples instead of the median of probabilities.
- Calibration: hook for a logit-linear map  p' = sigmoid(A * logit(p) + B). The 2026-10-05 backtest
  fitted A~0.94, B~-0.10, which made out-of-fold scores slightly WORSE, so it is left at identity.
  Refit once the bot has ~100+ resolved tournament forecasts of its own (see ../BACKTEST.md).
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime

# Backtest (BACKTEST.md) found no gain from calibrating, so identity. A > 1 extremizes, A < 1 shrinks.
# Identity (1.0, 0.0) means "no calibration".
CALIBRATION_A = 1.0
CALIBRATION_B = 0.0

# Hard bounds on a final binary forecast. Under a log score one confident miss is very
# expensive, so we never go past these.
P_MIN, P_MAX = 0.02, 0.98


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def clamp(p: float, lo: float = P_MIN, hi: float = P_MAX) -> float:
    return max(lo, min(hi, p))


######################################## PROMPTS ########################################


def template_binary_prompt(
    question_text: str,
    background: str,
    resolution_criteria: str,
    fine_print: str,
    research: str,
    today: str | None = None,
) -> str:
    """The Metaculus template's binary prompt, verbatim (FallTemplateBot2026). Used as the baseline."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    return f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{research}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


def improved_binary_prompt(
    question_text: str,
    background: str,
    resolution_criteria: str,
    fine_print: str,
    research: str,
    today: str | None = None,
) -> str:
    today = today or datetime.now().strftime("%Y-%m-%d")
    return f"""
You are an experienced superforecaster with a strong calibration record. You are scored with a log score,
so both overconfidence and needless hedging cost points.

Question:
{question_text}

Background:
{background}

Resolution criteria (not yet satisfied as of today):
{resolution_criteria}

{fine_print}

Research notes (may be incomplete; trust facts over opinions):
{research}

Today is {today}.

Work through these steps briefly:
1. Resolution mechanics: what exactly must happen, by when, and how many days are left. Note anything
   in the criteria that makes Yes easier or harder than the headline suggests.
2. Outside view: name a reference class and give a base rate for events like this in a window of this
   length. If the question asks whether a measured quantity will rise or exceed a level, think about its
   recent trend and normal volatility over the time left.
3. Inside view: the specific evidence that moves you away from the base rate, and how far.
4. Status quo and time: the world usually changes slowly. If the event needs something new to happen
   and little time is left, that pushes toward No; if the event is already in motion or scheduled,
   that pushes toward Yes.
5. Final check: would you be surprised to be wrong? Questions on forecasting platforms resolve No more
   often than people expect. Don't stay near 50% if the evidence clearly points one way, and don't
   go past 3% or 97% unless the outcome is nearly certain.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
""".strip()


######################################## PARSING ########################################

_PROB_RE = re.compile(r"Probability:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def parse_probability(text: str) -> float | None:
    """Last "Probability: ZZ%" in the text, as a decimal. None if absent."""
    matches = _PROB_RE.findall(text or "")
    if not matches:
        return None
    value = float(matches[-1]) / 100
    if not 0 <= value <= 1:
        return None
    return value


######################################## AGGREGATION ########################################


def aggregate_median(preds: list[float]) -> float:
    """The template's aggregation (median), with its 1%/99% clamp."""
    return clamp(float(statistics.median(preds)), 0.01, 0.99)


def aggregate_mean_logodds(preds: list[float]) -> float:
    return sigmoid(statistics.fmean(logit(clamp(p, 0.01, 0.99)) for p in preds))


def calibrate(p: float, a: float = CALIBRATION_A, b: float = CALIBRATION_B) -> float:
    return sigmoid(a * logit(p) + b)


def aggregate_binary(preds: list[float]) -> float:
    """What the live bot submits for a binary question."""
    return clamp(calibrate(aggregate_mean_logodds(preds)))


######################################## SCORING ########################################


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def log_score(p: float, outcome: int) -> float:
    """Natural-log score (higher is better, 0 is perfect)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p if outcome else 1 - p)


def fit_calibration(preds: list[float], outcomes: list[int]) -> tuple[float, float]:
    """Fit p' = sigmoid(A*logit(p)+B) by maximum likelihood (Newton's method, small ridge on B)."""
    a, b = 1.0, 0.0
    xs = [logit(clamp(p, 0.01, 0.99)) for p in preds]
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, outcomes):
            q = sigmoid(a * x + b)
            r = y - q
            w = q * (1 - q)
            ga += r * x
            gb += r
            haa += w * x * x
            hab += w * x
            hbb += w
        # ridge toward the identity map (A=1, B=0) to keep small samples sane
        lam = 1.0
        ga -= lam * (a - 1)
        gb -= lam * b
        haa += lam
        hbb += lam
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-8:
            break
    return a, b
