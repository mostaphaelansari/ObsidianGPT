from __future__ import annotations

import re
from pathlib import Path


_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

# Spans linkify_mentions must leave alone: fenced code, inline code, existing wikilinks.
# A fence closes only on a line with the same backtick/tilde run that opened it.
_PROTECTED = re.compile(
    r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$|`[^`\n]*`|\[\[[^\]]*\]\]",
    re.DOTALL | re.MULTILINE,
)


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
    """Wrap known note titles in `[[...]]` when they appear as whole words (conservative).

    Code (inline and fenced) and existing wikilinks are left untouched, and all
    titles are matched in one pass so a link is never nested inside another.
    """
    # Longer titles first to prefer multi-word matches
    titles = sorted(
        (t for t in known_titles if t.strip() and not _WIKILINK.search(t)),
        key=len,
        reverse=True,
    )
    if not titles:
        return text

    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in titles) + r")\b",
        flags=re.IGNORECASE,
    )

    def link_plain(segment: str) -> str:
        return pattern.sub(lambda m: f"[[{m.group(0)}]]", segment)

    out: list[str] = []
    pos = 0
    for m in _PROTECTED.finditer(text):
        out.append(link_plain(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(link_plain(text[pos:]))
    return "".join(out)


def extract_wikilinks(markdown: str) -> list[str]:
    return [m.strip() for m in _WIKILINK.findall(markdown)]
