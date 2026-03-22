from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SubTask:
    """One actionable sub-question derived from a user goal."""

    id: str
    question: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage from the vault used as evidence."""

    path: Path
    excerpt: str
    score: float


@dataclass
class Plan:
    goal: str
    subtasks: list[SubTask]


@dataclass
class Synthesis:
    """Structured output from reasoning over retrieved evidence."""

    goal: str
    markdown_body: str
    source_paths: list[Path] = field(default_factory=list)


@dataclass
class WriteResult:
    path: Path
    created: bool
