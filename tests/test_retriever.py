"""Tests for the chunked retriever."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from obsidiangpt.retriever import (
    _chunk_words,
    _cosine_bow,
    _read_body,
    _tokenize,
    retrieve,
)
from obsidiangpt.config import RunConfig
from obsidiangpt.models import Plan


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_short_tokens(self):
        # Tokens < 2 chars are excluded
        result = _tokenize("a bb ccc")
        assert "a" not in result
        assert "bb" in result

    def test_handles_empty(self):
        assert _tokenize("") == []


class TestChunkWords:
    def test_short_text_single_chunk(self):
        text = "word " * 100
        chunks = _chunk_words(text.strip())
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        # 900 words → at least 2 chunks at 400 words / 50-word stride
        text = " ".join(f"word{i}" for i in range(900))
        chunks = _chunk_words(text)
        assert len(chunks) >= 2

    def test_chunks_overlap(self):
        # The end of chunk 0 and start of chunk 1 should share words
        text = " ".join(str(i) for i in range(500))
        chunks = _chunk_words(text)
        c0_words = set(chunks[0].split())
        c1_words = set(chunks[1].split())
        assert len(c0_words & c1_words) > 0

    def test_empty_text(self):
        assert _chunk_words("") == []


class TestReadBody:
    def test_strips_frontmatter(self, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("---\ntitle: Test\n---\n\nBody text here.", encoding="utf-8")
        body = _read_body(note)
        assert "title:" not in body
        assert "Body text here." in body

    def test_no_frontmatter(self, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("Just body.", encoding="utf-8")
        assert _read_body(note) == "Just body."

    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_body(tmp_path / "missing.md") == ""


class TestCosineBow:
    def test_identical_vectors_score_one(self):
        q = {"mqtt": 2.0, "broker": 1.0}
        tokens = ["mqtt", "broker", "mqtt"]
        score = _cosine_bow(q, tokens)
        assert abs(score - 1.0) < 1e-9

    def test_zero_overlap_score_zero(self):
        q = {"mqtt": 1.0}
        tokens = ["postgres", "database"]
        assert _cosine_bow(q, tokens) == 0.0

    def test_partial_overlap(self):
        q = {"mqtt": 1.0, "tls": 1.0}
        tokens = ["mqtt", "tcp", "ip"]
        score = _cosine_bow(q, tokens)
        assert 0 < score < 1.0


# ---------------------------------------------------------------------------
# Integration: retrieve() against a temp vault
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_vault(tmp_path):
    """A mini vault with two notes."""
    (tmp_path / "mqtt.md").write_text(
        "# MQTT\nMQTT is a lightweight pub/sub messaging protocol. "
        "It uses a broker to route messages between clients.",
        encoding="utf-8",
    )
    (tmp_path / "postgres.md").write_text(
        "# PostgreSQL\nPostgreSQL is a relational database. "
        "It supports SQL and ACID transactions.",
        encoding="utf-8",
    )
    return tmp_path


class TestRetrieve:
    def test_returns_most_relevant_first(self, tiny_vault):
        cfg = RunConfig(vault_root=tiny_vault)
        p = Plan(goal="MQTT broker messages", subtasks=[])
        chunks = retrieve(cfg, p)
        assert chunks, "Should return at least one chunk"
        # The MQTT note should score higher than postgres
        assert "mqtt" in chunks[0].path.stem.lower()

    def test_top_k_respected(self, tiny_vault):
        cfg = RunConfig(vault_root=tiny_vault, top_k_chunks=1)
        p = Plan(goal="MQTT broker", subtasks=[])
        chunks = retrieve(cfg, p)
        assert len(chunks) <= 1

    def test_empty_query_returns_empty(self, tiny_vault):
        cfg = RunConfig(vault_root=tiny_vault)
        p = Plan(goal="", subtasks=[])
        assert retrieve(cfg, p) == []

    def test_no_match_returns_empty(self, tiny_vault):
        cfg = RunConfig(vault_root=tiny_vault)
        # Query with tokens that don't appear in either note
        p = Plan(goal="xyzzy quux zork", subtasks=[])
        result = retrieve(cfg, p)
        assert result == []

    def test_scores_are_positive(self, tiny_vault):
        cfg = RunConfig(vault_root=tiny_vault)
        p = Plan(goal="database SQL", subtasks=[])
        chunks = retrieve(cfg, p)
        for c in chunks:
            assert c.score > 0
