"""Retriever: scores vault notes by keyword overlap with the query.

Improvement over the original: instead of scoring the entire file, each
.md file is split into overlapping word-based chunks (≤400 words with a
50-word stride).  Chunks are scored independently so a highly relevant
paragraph inside a long note isn't drowned out.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig
from .models import RetrievedChunk, Plan

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

_CHUNK_WORDS = 400   # target chunk size in words
_STRIDE_WORDS = 50   # overlap between successive chunks
_MIN_SCORE = 0.08    # discard chunks below this cosine similarity threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "what", "how", "why", "who", "when", "where", "does", "do", "did",
    "can", "could", "would", "should", "has", "have", "had", "which", "about",
    "from", "your", "my", "our", "we", "you", "he", "she", "it", "they", "me",
    "him", "her", "us", "them", "i", "am", "are", "was", "were", "be", "been",
    "some", "any", "all", "most", "many", "much", "very"
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]{2,}", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _iter_markdown_files(root: Path, max_files: int) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.md"):
        if ".obsidian" in p.parts:
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def _read_body(path: Path) -> str:
    """Read a markdown file, stripping YAML front-matter."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            raw = raw[end + 4:]
    return raw.strip()


@dataclass
class _ScoredChunk:
    score: float
    path: Path
    excerpt: str
    chunk_index: int


def _chunk_words(text: str) -> list[str]:
    """Split *text* into overlapping word windows."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + _CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += _CHUNK_WORDS - _STRIDE_WORDS
    return chunks


def _cosine_bow(q_weights: dict[str, float], doc_tokens: list[str]) -> float:
    """Bag-of-words cosine similarity with weighted query tokens."""
    d_counts: Counter[str] = Counter(doc_tokens)
    dot = sum(q_weights.get(t, 0.0) * d_counts.get(t, 0) for t in q_weights)
    norm_q = sum(w * w for w in q_weights.values()) ** 0.5
    norm_d = sum(c * c for c in d_counts.values()) ** 0.5
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    cfg: RunConfig,
    plan_obj: Plan,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k most relevant chunks from the vault.

    Words from the original goal are weighted 10x higher than words from the sub-questions
    so that generic terms like 'structure' don't overwhelm core concepts like 'SQL'.
    """
    k = top_k if top_k is not None else cfg.top_k_chunks
    
    goal_tokens = _tokenize(plan_obj.goal)
    subtask_tokens = []
    for st in plan_obj.subtasks:
        subtask_tokens.extend(_tokenize(st.question))
        
    if not goal_tokens and not subtask_tokens:
        return []

    # Build heavily weighted query vector: 10 points for goal words, 1 for subtask words
    q_weights: dict[str, float] = {}
    for t in subtask_tokens:
        q_weights[t] = q_weights.get(t, 0.0) + 1.0
    for t in goal_tokens:
        q_weights[t] = q_weights.get(t, 0.0) + 10.0
    scored: list[_ScoredChunk] = []

    for path in _iter_markdown_files(cfg.vault_root, cfg.max_files):
        body = _read_body(path)
        if not body:
            continue
        for i, chunk_text in enumerate(_chunk_words(body)):
            doc_tokens = _tokenize(chunk_text)
            if not doc_tokens:
                continue
            score = _cosine_bow(q_weights, doc_tokens)
            if score >= _MIN_SCORE:
                scored.append(_ScoredChunk(
                    score=score,
                    path=path,
                    excerpt=chunk_text,
                    chunk_index=i,
                ))

    scored.sort(key=lambda x: x.score, reverse=True)
    # De-duplicate: keep at most one chunk per file (highest-scoring)
    seen: set[Path] = set()
    results: list[RetrievedChunk] = []
    for sc in scored:
        if sc.path not in seen:
            seen.add(sc.path)
            results.append(RetrievedChunk(path=sc.path, excerpt=sc.excerpt, score=sc.score))
        if len(results) >= k:
            break
    return results
