"""Reasoner: synthesises a coherent answer from a plan and evidence chunks.

Uses the active LLM (cloud provider or Ollama, see ``llm.py``); falls back
to a structured template when no LLM is reachable.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

from .llm import LLMError, call_llm
from .models import Plan, RetrievedChunk, Synthesis

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are ObsidianGPT — a precise, intellectually honest second-brain research agent operating directly across the user's local Obsidian vault.

**Core Rules — Never break these:**
• **Zero Hallucination:** You only know what exists in the provided context (the retrieved note chunks from the vault). Never invent facts, note titles, dates, references, people, papers, URLs, code behavior, or vault structure. 
• If the retrieved chunks do not contain enough information to accomplish the user's goal, say explicitly: "I don't have enough information in the retrieved vault notes to fully answer this goal. Could you provide a different query or add more notes to your vault?"
• **High Signal, Zero Fluff:** Prefer dense, high-signal, academic-grade answers. Eliminate all conversational filler phrases. Just output the information directly.

**Markdown & Formatting Standards:**
• Always write clean, correct Obsidian-flavored Markdown.
• Explicitly cite your sources using realistic Obsidian wikilinks: `[[Note Title]]` whenever referencing retrieved chunks. 
• Generously use headers (`##`, `###`), bullet points (`-`), and tables to logically structure the synthesized information.
• Preserve formatting standards: use `#tags` for themes, `- [ ]` for tasks, `**bold**` for emphasis, `code blocks` for syntax, and `$` / `$$` for mathematical equations.

**Output Structure & Capabilities:**
When processing the user's specific Research Goal, you must:
1.  **Synthesize & Summarize:** Merge the raw retrieved chunks into a cohesive, nuanced answer. Do not just copy-paste; synthesize the concepts together.
2.  **Highlight Contradictions:** If the retrieved notes contain conflicting claims or outdated information, explicitly list these contradictions first.
3.  **Actionable Conclusion:** End your synthesis with a distinct section titled **"Next Steps / Potential New Notes"**. Provide 2–4 focused bullet points suggesting unresolved questions, logical follow-up tasks, or titles for prospective new atomic notes to expand the user's knowledge graph.

**Signature:**
Always sign the very bottom of your generated synthesis with:  
— ObsidianGPT
{current_date}
"""


def _build_context(plan: Plan, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user turn sent to the LLM."""
    lines: list[str] = [
        f"**Goal:** {plan.goal}",
        "",
        "**Sub-questions:**",
    ]
    for st in plan.subtasks:
        lines.append(f"- {st.question}")
    lines.append("")

    if chunks:
        lines.append("**Retrieved excerpts:**")
        lines.append("")
        for c in chunks:
            rel = c.path.as_posix()
            lines.append(f"---\n*Source: `{rel}` (score {c.score:.3f})*")
            lines.append("")
            lines.append(c.excerpt)
            lines.append("")
    else:
        lines.append("*No relevant notes were found in the vault.*")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _template_synthesis(plan: Plan, chunks: list[RetrievedChunk]) -> str:
    """Build a template-based markdown body (no LLM)."""
    sources = sorted({c.path for c in chunks}, key=lambda p: str(p))
    secs: list[str] = [
        "## Goal",
        plan.goal,
        "",
        "## Sub-questions",
    ]
    for st in plan.subtasks:
        secs.append(f"- {st.question}")
    secs.append("")

    if chunks:
        secs.append("## Evidence from vault")
        secs.append("")
        for c in chunks:
            rel = c.path.as_posix()
            secs.append(f"### `{rel}` _(score {c.score:.3f})_")
            secs.append("")
            # Excerpts are cut mid-note, so their markdown/math may be unbalanced:
            # fence them (longer than any backtick run inside) to show them verbatim.
            longest_run = max((len(r) for r in re.findall(r"`+", c.excerpt)), default=0)
            fence = "`" * max(3, longest_run + 1)
            secs.extend([f"{fence}text", c.excerpt, fence, ""])
    else:
        secs.extend([
            "## Evidence from vault",
            "_No matching notes found. Add notes or broaden the goal._",
            "",
        ])

    secs.extend([
        "## Synthesis",
        "> [!note]",
        "> This section is template-generated (offline mode). Start Ollama or set a provider API key to enable AI synthesis.",
        "",
        "## Contradictions & Gaps",
        "- (None detected automatically — add manual review or enable AI mode.)",
        "",
        "## Sources",
    ])
    for p in sources:
        secs.append(f"- `{p.as_posix()}`")
    secs.append("")

    return "\n".join(secs).strip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason(
    plan: Plan,
    chunks: list[RetrievedChunk],
    *,
    model: str | None = None,
) -> Synthesis:
    """Synthesise an answer from *plan* and evidence *chunks*.

    Uses the active LLM; falls back to a structured template when no LLM
    is reachable.
    """
    sources = sorted({c.path for c in chunks}, key=lambda p: str(p))

    context = _build_context(plan, chunks)
    llm_body = ""
    try:
        current_date_str = datetime.date.today().strftime("%Y-%m-%d")
        llm_body = call_llm(
            system_prompt=_SYSTEM_PROMPT.format(current_date=current_date_str),
            user_prompt=context,
            model=model,
        )
    except LLMError as exc:
        logger.warning("LLM call failed, using template fallback: %s", exc)

    if llm_body and llm_body.strip():
        body = llm_body.strip() + "\n"
    else:
        body = _template_synthesis(plan, chunks)

    return Synthesis(goal=plan.goal, markdown_body=body, source_paths=sources)
