"""Provider-mode selection in gigbot_bot (imports forecasting_tools, so a few seconds)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("METACULUS_TOKEN", "offline-test")
import gigbot_bot as g  # noqa: E402


def test_openrouter_key_wins_over_gemini():
    assert g.provider_mode({"OPENROUTER_API_KEY": "a", "GEMINI_API_KEY": "b"}) == "openrouter"


def test_gemini_is_the_interim_when_only_its_key_exists():
    assert g.provider_mode({"GEMINI_API_KEY": "b"}) == "gemini"
    assert g.provider_mode({"GEMINI_API_KEY": "b", "OPENROUTER_API_KEY": ""}) == "gemini"


def test_no_key_keeps_openrouter_defaults_so_the_gate_skips():
    assert g.provider_mode({}) == "openrouter"


def test_gemini_defaults_are_free_tier_and_research_free():
    d = g.DEFAULTS["gemini"]
    assert all(m.startswith("gemini/") and "lite" in m for m in d["FORECASTER_MODELS"].split(","))
    assert d["RESEARCHER_MODEL"] == "none"
    assert int(d["PREDICTIONS_PER_QUESTION"]) <= 3


def test_make_llm_wraps_only_gemini():
    assert isinstance(g.make_llm("gemini/gemini-3.1-flash-lite", temperature=0.1), g.GeminiLlm)
    assert not isinstance(g.make_llm("openrouter/openai/gpt-6-luna", temperature=0.1), g.GeminiLlm)
