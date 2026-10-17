"""Config: runtime options for an ObsidianGPT run."""
from __future__ import annotations

from dataclasses import dataclass
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

    model: str | None = None
    """Model for the active LLM provider; None uses that provider's *_MODEL env var or default."""

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.vault_root / "_obsidiangpt"
