"""Writer: persists a Synthesis as a markdown note in the vault output dir."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import Synthesis, WriteResult


def _slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80] or "note"


def write_synthesis_note(
    synthesis: Synthesis,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> WriteResult:
    """Write *synthesis* to a dated markdown file inside *out_dir*.

    The produced note includes:
    - YAML front-matter (title, created timestamp, tags)
    - The LLM/template-generated body
    - A ``## Related Notes`` section with wikilinks to every source note
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(synthesis.goal)
    path = out_dir / f"{stamp}-{slug}.md"

    # ── Front-matter ────────────────────────────────────────────────────────
    frontmatter_lines = [
        "---",
        f'title: "Agent run: {synthesis.goal[:120]}"',
        f"created: {datetime.now(timezone.utc).isoformat()}",
        "tags:",
        "  - obsidiangpt/generated",
        "---",
        "",
    ]

    # ── AI disclaimer callout ────────────────────────────────────────────────
    disclaimer = (
        "> [!note] AI-generated\n"
        "> This note was produced by ObsidianGPT. Verify claims against primary sources.\n\n"
    )

    # ── Related notes backlinks ──────────────────────────────────────────────
    related_lines: list[str] = []
    if synthesis.source_paths:
        related_lines.append("\n## Related Notes\n")
        for p in synthesis.source_paths:
            related_lines.append(f"- [[{p.stem}]]")
        related_lines.append("")

    content = (
        "\n".join(frontmatter_lines)
        + disclaimer
        + synthesis.markdown_body
        + ("\n" + "\n".join(related_lines) if related_lines else "")
    )

    if dry_run:
        return WriteResult(path=path, created=False)

    path.write_text(content, encoding="utf-8")
    return WriteResult(path=path, created=True)
