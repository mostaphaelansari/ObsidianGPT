from __future__ import annotations

import re
from pathlib import Path


_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


def list_note_titles(vault_root: Path, *, max_files: int = 5000) -> set[str]:
    """Basenames without `.md` for possible `[[wikilink]]` targets."""
    titles: set[str] = set()
    n = 0
    for p in vault_root.rglob("*.md"):
        if ".obsidian" in p.parts:
            continue
        titles.add(p.stem)
        n += 1
        if n >= max_files:
            break
    return titles


def linkify_mentions(text: str, known_titles: set[str]) -> str:
    """Wrap known note titles in `[[...]]` when they appear as whole words (conservative)."""
    if not known_titles:
        return text

    # Longer titles first to prefer multi-word matches
    sorted_titles = sorted(known_titles, key=len, reverse=True)
    out = text
    for title in sorted_titles:
        if not title.strip():
            continue
        if _WIKILINK.search(title):
            continue
        # Skip if already linked
        pattern = re.compile(
            r"(?<!\[)\b" + re.escape(title) + r"\b(?!\])",
            flags=re.IGNORECASE,
        )

        def repl(m: re.Match[str]) -> str:
            matched = m.group(0)
            return f"[[{matched}]]"

        out = pattern.sub(repl, out)
    return out


def extract_wikilinks(markdown: str) -> list[str]:
    return [m.strip() for m in _WIKILINK.findall(markdown)]
