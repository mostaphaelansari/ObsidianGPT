"""Planner: breaks a user goal into actionable sub-questions.

LLM-backed (cloud provider or Ollama, see ``llm.py``); falls back to
deterministic rule-based splitting when no LLM is reachable.
"""
from __future__ import annotations

import logging
import re
import uuid

logger = logging.getLogger(__name__)

from .llm import LLMError, call_llm
from .models import Plan, SubTask

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research planner for a personal knowledge management system.
Given a user's high-level goal, decompose it into 3–6 focused sub-questions
that together, when answered, fully address the goal.

Rules:
- Output ONLY a numbered list. No preamble, no commentary.
- Each item must be a single, self-contained question ending with "?".
- Be specific and concrete — avoid vague or overlapping questions.

Example output:
1. What are the core components of an MQTT broker?
2. How does TLS encryption work with MQTT?
3. How can PostgreSQL be used to persist MQTT messages?
"""


def _parse_numbered_list(text: str) -> list[str]:
    """Extract question strings from a numbered list response."""
    results: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.)]\s+(.+)", line)
        if m:
            q = m.group(1).strip()
            if q:
                results.append(q)
    return results


# ---------------------------------------------------------------------------
# Deterministic fallback (no LLM)
# ---------------------------------------------------------------------------

def _split_goal(goal: str) -> list[str]:
    """Produce sub-questions from the goal text using simple heuristics."""
    text = goal.strip()
    if not text:
        return []

    # Prefer explicit numbered or bulleted lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        cleaned: list[str] = []
        for ln in lines:
            ln = re.sub(r"^[-*]\s+", "", ln)
            ln = re.sub(r"^\d+[.)]\s+", "", ln)
            cleaned.append(ln)
        return cleaned

    # Split on strong delimiters
    parts = re.split(r"\s*(?:;|\.(?:\s+|$)|\n|(?:\s+and\s+))\s*", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts

    # Single goal: derive a small default plan
    return [
        f"Clarify scope and success criteria for: {text}",
        f"Identify constraints, risks, and dependencies for: {text}",
        f"Outline concrete steps or architecture for: {text}",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(goal: str, *, model: str | None = None) -> Plan:
    """Return a :class:`Plan` decomposing *goal* into sub-questions.

    Tries the active LLM first; falls back to rule-based splitting when no
    LLM is reachable or the response cannot be parsed.
    """
    goal = goal.strip()
    questions: list[str] = []

    llm_response = ""
    try:
        llm_response = call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Goal: {goal}",
            model=model,
        )
    except LLMError as exc:
        logger.warning("LLM call failed, using offline fallback: %s", exc)
    if llm_response:
        questions = _parse_numbered_list(llm_response)

    if not questions:
        questions = _split_goal(goal)

    subtasks = [SubTask(id=str(uuid.uuid4())[:8], question=q) for q in questions]
    return Plan(goal=goal, subtasks=subtasks)
