"""LLM backend for ObsidianGPT — Ollama (primary) with Gemini fallback.

Priority
--------
1. Ollama  — if OLLAMA_MODEL is set (or defaults to qwen3:1.7b) and the
             Ollama server is reachable at OLLAMA_HOST.
2. Gemini  — if GEMINI_API_KEY is set.
3. Offline — returns "" so callers fall back to deterministic mode.

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


class LLMError(RuntimeError):
    """Raised when all backends fail."""


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
# Gemini backend (kept as cloud fallback)
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
        raise LLMError("google-genai not installed. Run: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "qwen3:1.7b",
) -> str:
    """Call the best available LLM backend and return the text reply.

    Returns "" in offline mode (no backends configured) so callers can
    fall back gracefully without crashing.
    """
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", model).strip()
    gemini_key   = os.environ.get("GEMINI_API_KEY", "").strip()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()

    # ── 1. Ollama (local, preferred) ────────────────────────────────────────
    if ollama_model:
        try:
            logger.debug("Calling Ollama: model=%s host=%s", ollama_model, ollama_host)
            return _call_ollama(system_prompt, user_prompt, ollama_model, ollama_host)
        except LLMError as exc:
            logger.warning("Ollama failed, trying Gemini fallback: %s", exc)

    # ── 2. Gemini (cloud fallback) ───────────────────────────────────────────
    if gemini_key:
        try:
            logger.debug("Calling Gemini: model=%s", gemini_model)
            return _call_gemini(system_prompt, user_prompt, gemini_model, gemini_key)
        except LLMError as exc:
            logger.warning("Gemini also failed, using offline mode: %s", exc)

    # ── 3. Offline ───────────────────────────────────────────────────────────
    logger.debug("No LLM backend available — offline fallback mode.")
    return ""
