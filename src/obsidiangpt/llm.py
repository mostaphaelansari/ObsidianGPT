"""LLM backend for ObsidianGPT — a cloud provider when an API key is set, Ollama otherwise.

Provider selection
------------------
1. ``LLM_PROVIDER`` — force one of: anthropic, openai, gemini, ollama.
2. Otherwise the first API key found wins:
   ``ANTHROPIC_API_KEY`` → anthropic, ``OPENAI_API_KEY`` → openai,
   ``GEMINI_API_KEY`` → gemini.
3. No API key → local Ollama at ``OLLAMA_HOST`` (default model qwen3:1.7b).

If a cloud provider fails, the call falls back to Ollama; if Ollama is not
reachable either, it returns "" so callers use their deterministic offline mode.

Each provider's model can be set with ``ANTHROPIC_MODEL``, ``OPENAI_MODEL``,
``GEMINI_MODEL`` or ``OLLAMA_MODEL``. ``OPENAI_BASE_URL`` points the OpenAI
backend at any OpenAI-compatible API (OpenRouter, Groq, Together, LM Studio...).

Usage
-----
    from obsidiangpt.llm import call_llm
    text = call_llm("You are a planner.", "Break down: design an MQTT system")
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 20

PROVIDERS = ("anthropic", "openai", "gemini", "ollama")

# Checked in this order when LLM_PROVIDER is not set.
_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_MODEL_ENV = {
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "ollama": "OLLAMA_MODEL",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:1.7b",
}

# Claude models that accept server-side refusal fallbacks and an effort level.
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    """Raised when a backend fails."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------

def active_provider() -> str:
    """Return the provider ``call_llm`` will try first."""
    forced = _env("LLM_PROVIDER").lower()
    if forced in PROVIDERS:
        return forced
    if forced:
        logger.warning("Unknown LLM_PROVIDER=%r (expected one of %s); auto-detecting.", forced, ", ".join(PROVIDERS))

    for provider, key_env in _API_KEY_ENV.items():
        if _env(key_env):
            return provider
    return "ollama"


def resolve_model(provider: str, model: str | None = None) -> str:
    """Explicit *model*, else the provider's ``*_MODEL`` env var, else its default."""
    return (model or "").strip() or _env(_MODEL_ENV[provider]) or _DEFAULT_MODELS[provider]


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str,
    host: str,
) -> str:
    """Call the local Ollama server.  Raises LLMError on failure."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            return body["message"]["content"]
    except urllib.error.URLError as exc:
        raise LLMError(f"Ollama request failed ({url}): {exc}") from exc
    except (KeyError, json.JSONDecodeError) as exc:
        raise LLMError(f"Unexpected Ollama response: {exc}") from exc


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Call the Claude API.  Raises LLMError on failure."""
    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise LLMError("anthropic not installed. Run: pip install -e .[anthropic]") from exc
    if int(anthropic.__version__.split(".")[0]) < 1:
        raise LLMError(f"anthropic {anthropic.__version__} is too old. Run: pip install -U 'anthropic>=1.0'")

    extra: dict = {}
    if model in _CLAUDE_FALLBACK_MODELS:
        # If a safety classifier declines, the API re-runs the request on a
        # fallback Claude model inside the same call.
        extra = {
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "output_config": {"effort": "medium"},
        }

    try:
        client = anthropic.Anthropic(api_key=api_key or None)
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **extra,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError(f"Anthropic rejected the API key: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined the request (stop_reason=refusal)")
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------------------
# OpenAI (and OpenAI-compatible) backend
# ---------------------------------------------------------------------------

def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Call OpenAI, or any OpenAI-compatible API via OPENAI_BASE_URL.  Raises LLMError on failure."""
    try:
        import openai  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise LLMError("openai not installed. Run: pip install -e .[openai]") from exc

    try:
        client = openai.OpenAI(api_key=api_key or None)  # reads OPENAI_BASE_URL itself
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
    except openai.OpenAIError as exc:
        raise LLMError(f"OpenAI request failed: {exc}") from exc

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _parse_retry_delay(exc_str: str) -> float:
    m = re.search(r"'retryDelay':\s*'(\d+)s'", exc_str)
    return float(m.group(1)) if m else _DEFAULT_RETRY_DELAY


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Call the Gemini API with retry on rate-limit.  Raises LLMError on failure."""
    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import types  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise LLMError("google-genai not installed. Run: pip install -e .[gemini]") from exc

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            is_rate_limit = any(k in exc_str for k in ("429", "retryDelay", "RESOURCE_EXHAUSTED"))
            if is_rate_limit and attempt < _MAX_RETRIES:
                delay = _parse_retry_delay(exc_str)
                logger.warning("Gemini rate limit (attempt %d/%d) — waiting %.0fs", attempt, _MAX_RETRIES, delay)
                print(f"  [rate-limit] waiting {delay:.0f}s before retry ({attempt}/{_MAX_RETRIES})...")
                time.sleep(delay)
            else:
                break

    raise LLMError(f"Gemini failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc  # type: ignore[misc]


_CLOUD_BACKENDS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
) -> str:
    """Call the active LLM provider and return the text reply.

    *model* overrides the active provider's model. Returns "" in offline mode
    (no provider reachable) so callers can fall back without crashing.
    """
    provider = active_provider()

    # ── 1. Cloud provider (when an API key is configured) ───────────────────
    if provider != "ollama":
        cloud_model = resolve_model(provider, model)
        try:
            logger.debug("Calling %s: model=%s", provider, cloud_model)
            return _CLOUD_BACKENDS[provider](system_prompt, user_prompt, cloud_model, _env(_API_KEY_ENV[provider]))
        except LLMError as exc:
            logger.warning("%s failed, falling back to Ollama: %s", provider, exc)
        model = None  # a cloud model name means nothing to Ollama

    # ── 2. Ollama (local) ────────────────────────────────────────────────────
    ollama_model = resolve_model("ollama", model)
    ollama_host = _env("OLLAMA_HOST") or "http://localhost:11434"
    try:
        logger.debug("Calling Ollama: model=%s host=%s", ollama_model, ollama_host)
        return _call_ollama(system_prompt, user_prompt, ollama_model, ollama_host)
    except LLMError as exc:
        logger.warning("Ollama failed, using offline mode: %s", exc)

    # ── 3. Offline ───────────────────────────────────────────────────────────
    return ""
