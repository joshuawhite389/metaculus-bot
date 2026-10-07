"""
Which API key does each configured model need, and is it set?

gigbot's bot used to skip a run only when an `openrouter/` model had no OPENROUTER_API_KEY. This
generalises that: every model name (litellm / forecasting-tools style) maps to the environment
variable(s) its provider needs, so changing FORECASTER_MODELS etc. through repository variables
to any provider "just works" the moment its key is set, and skips quietly (with a clear reason)
until then.

`metaculus/...` models are a special case. forecasting-tools routes them to the Metaculus LLM
proxy at llm-proxy.metaculus.com using METACULUS_TOKEN, but that hostname no longer exists
(NXDOMAIN on public resolvers, checked 2026-10-07), so those models can never work and are
reported as such instead of failing every run with a connection error.

Pure functions, no network, no imports beyond the standard library: see tests/test_provider_keys.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

PROXY_HOST = "llm-proxy.metaculus.com"
PROXY_GONE_REASON = (
    f"routes to the Metaculus LLM proxy ({PROXY_HOST}), which no longer exists "
    "(hostname has no DNS record; checked 2026-10-07). Use an OpenRouter or provider key instead."
)

# Longest prefix wins. Values: the env var(s) that must ALL be set (any one group suffices when
# the value is a list of groups).
_PREFIX_KEYS: dict[str, list[tuple[str, ...]]] = {
    "openrouter/": [("OPENROUTER_API_KEY",)],
    "openai/": [("OPENAI_API_KEY",)],
    "anthropic/": [("ANTHROPIC_API_KEY",)],
    "gemini/": [("GEMINI_API_KEY",), ("GOOGLE_API_KEY",)],
    "vertex_ai/": [("GOOGLE_APPLICATION_CREDENTIALS",)],
    "perplexity/": [("PERPLEXITY_API_KEY",)],
    "exa/": [("EXA_API_KEY",)],
    "smart-searcher/": [("EXA_API_KEY",)],
    "asknews/": [("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"), ("ASKNEWS_API_KEY",)],
    "xai/": [("XAI_API_KEY",)],
    "deepseek/": [("DEEPSEEK_API_KEY",)],
    "mistral/": [("MISTRAL_API_KEY",)],
    "groq/": [("GROQ_API_KEY",)],
}

# Bare model names (no provider prefix) that litellm sends to OpenAI or Anthropic directly.
_BARE_OPENAI = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_BARE_ANTHROPIC = ("claude-",)


def _is_set(env: Mapping[str, str], name: str) -> bool:
    val = env.get(name)
    return bool(val and val.strip())


def required_keys(model: str) -> list[tuple[str, ...]] | None:
    """
    Alternative key groups for `model`, any one of which is enough. `None` means "unknown
    provider, don't gate on it". An empty list means the model can never work (dead proxy).
    """
    model = model.strip()
    if not model:
        return None
    if model.startswith("metaculus/"):
        return []
    for prefix in sorted(_PREFIX_KEYS, key=len, reverse=True):
        if model.startswith(prefix):
            return _PREFIX_KEYS[prefix]
    if "/" not in model:
        if model.startswith(_BARE_ANTHROPIC):
            return [("ANTHROPIC_API_KEY",)]
        if model.startswith(_BARE_OPENAI):
            return [("OPENAI_API_KEY",)]
    return None


def blocker_for(model: str, env: Mapping[str, str]) -> str | None:
    """A one-line reason this model cannot run now, or None if it can (or we can't tell)."""
    groups = required_keys(model)
    if groups is None:
        return None
    if groups == []:
        return f"{model}: {PROXY_GONE_REASON}"
    for group in groups:
        if all(_is_set(env, k) for k in group):
            return None
    wanted = " or ".join("+".join(g) for g in groups)
    return f"{model}: {wanted} is not set"


def run_blockers(models: Iterable[str], env: Mapping[str, str]) -> list[str]:
    """Deduplicated reasons the configured models cannot run, in input order. Empty = go."""
    seen: list[str] = []
    for m in models:
        reason = blocker_for(m, env)
        if reason and reason not in seen:
            seen.append(reason)
    return seen
