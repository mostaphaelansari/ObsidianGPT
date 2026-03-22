"""Config: runtime options for an ObsidianGPT run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Runtime options for a vault run."""

    vault_root: Path
    """Root folder of the Obsidian vault (contains notes and usually `.obsidian`)."""

    output_dir: Path | None = None
    """Where new notes are written; defaults to `<vault>/_obsidiangpt` if None."""

    dry_run: bool = False
    """If True, do not write files; print intended path and content preview."""

    max_files: int = 2000
    """Safety cap when scanning the vault."""

    top_k_chunks: int = 12
    """How many evidence chunks to pass to the reasoner."""

    gemini_model: str = field(default_factory=lambda: __import__("os").environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    """Gemini model identifier. Reads GEMINI_MODEL env var, falls back to gemini-2.0-flash."""

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.vault_root / "_obsidiangpt"
