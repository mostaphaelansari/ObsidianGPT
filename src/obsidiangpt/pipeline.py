"""Pipeline: orchestrates the full goal → note workflow."""
from __future__ import annotations

from .config import RunConfig
from .linker import linkify_mentions, list_note_titles
from .models import Plan, RetrievedChunk, Synthesis, WriteResult
from .planner import plan
from .reasoner import reason
from .retriever import retrieve
from .writer import write_synthesis_note


def run_goal(cfg: RunConfig, goal: str) -> tuple[Plan, list[RetrievedChunk], Synthesis, WriteResult]:
    """Execute the full pipeline for *goal*.

    Steps
    -----
    1. **Plan** — decompose the goal into sub-questions (LLM or heuristic).
    2. **Retrieve** — find the most relevant note chunks in the vault.
    3. **Reason** — synthesise an answer from the evidence (LLM or template).
    4. **Link** — inject wikilinks for known note titles.
    5. **Write** — persist the synthesis as a new markdown note.
    """
    p = plan(goal, model=cfg.model)
    chunks = retrieve(cfg, p)
    synthesis = reason(p, chunks, model=cfg.model)
    titles = list_note_titles(cfg.vault_root)
    synthesis.markdown_body = linkify_mentions(synthesis.markdown_body, titles)
    result = write_synthesis_note(
        synthesis,
        cfg.resolved_output_dir(),
        dry_run=cfg.dry_run,
    )
    return p, chunks, synthesis, result
