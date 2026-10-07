"""Unit tests for provider_keys (no network). Run: .venv/bin/python -m pytest tests/test_provider_keys.py -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provider_keys as pk  # noqa: E402

ENSEMBLE = ["openrouter/anthropic/claude-sonnet-5.5", "openrouter/openai/gpt-6.1-sol"]
RESEARCHER = "openrouter/openai/gpt-6.1-sol:online"
PARSER = "openrouter/openai/gpt-6-luna"
ALL = ENSEMBLE + [RESEARCHER, PARSER]


def test_openrouter_models_skip_without_key_and_run_with_it():
    assert pk.run_blockers(ALL, {}) == [
        f"{m}: OPENROUTER_API_KEY is not set" for m in ALL
    ]
    assert pk.run_blockers(ALL, {"OPENROUTER_API_KEY": "sk-or-test"}) == []


def test_empty_github_variable_counts_as_unset():
    assert pk.run_blockers(ALL, {"OPENROUTER_API_KEY": ""}) != []
    assert pk.run_blockers(ALL, {"OPENROUTER_API_KEY": "   "}) != []


def test_metaculus_proxy_models_are_always_blocked_even_with_token():
    env = {"METACULUS_TOKEN": "abc", "OPENROUTER_API_KEY": "x"}
    [reason] = pk.run_blockers(["metaculus/gpt-4o"], env)
    assert reason.startswith("metaculus/gpt-4o: ")
    assert pk.PROXY_HOST in reason and "no longer exists" in reason


def test_other_providers_map_to_their_keys():
    assert pk.blocker_for("gemini/gemini-2.5-flash", {}) == (
        "gemini/gemini-2.5-flash: GEMINI_API_KEY or GOOGLE_API_KEY is not set"
    )
    assert pk.blocker_for("gemini/gemini-2.5-flash", {"GOOGLE_API_KEY": "k"}) is None
    assert pk.blocker_for("anthropic/claude-sonnet-5.5", {"ANTHROPIC_API_KEY": "k"}) is None
    assert pk.blocker_for("claude-sonnet-5.5", {}) == "claude-sonnet-5.5: ANTHROPIC_API_KEY is not set"
    assert pk.blocker_for("gpt-4o", {}) == "gpt-4o: OPENAI_API_KEY is not set"
    assert pk.blocker_for("openai/gpt-4o-search-preview", {"OPENAI_API_KEY": "k"}) is None
    assert pk.blocker_for("perplexity/sonar-pro", {}) == "perplexity/sonar-pro: PERPLEXITY_API_KEY is not set"
    assert pk.blocker_for("asknews/news-summaries", {"ASKNEWS_CLIENT_ID": "a"}) is not None
    assert pk.blocker_for("asknews/news-summaries", {"ASKNEWS_CLIENT_ID": "a", "ASKNEWS_SECRET": "b"}) is None
    assert pk.blocker_for("asknews/news-summaries", {"ASKNEWS_API_KEY": "k"}) is None


def test_openrouter_free_models_only_need_the_openrouter_key():
    free = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert pk.blocker_for(free, {}) is not None
    assert pk.blocker_for(free, {"OPENROUTER_API_KEY": "k"}) is None


def test_unknown_provider_is_not_gated():
    assert pk.required_keys("somevendor/some-model") is None
    assert pk.blocker_for("somevendor/some-model", {}) is None
    assert pk.blocker_for("", {}) is None


def test_blockers_are_deduplicated_and_ordered():
    env = {}
    out = pk.run_blockers(["openrouter/a", "openrouter/a", "gemini/b"], env)
    assert out == ["openrouter/a: OPENROUTER_API_KEY is not set", "gemini/b: GEMINI_API_KEY or GOOGLE_API_KEY is not set"]
