"""Tests for the writer module."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsidiangpt.models import Synthesis, WriteResult
from obsidiangpt.writer import _slugify, write_synthesis_note


class TestSlugify:
    def test_lowercases_and_replaces_spaces(self):
        assert _slugify("Hello World") == "hello-world"

    def test_strips_special_chars(self):
        assert _slugify("MQTT: TLS & Postgres!") == "mqtt-tls-postgres"

    def test_max_length(self):
        long = "a" * 200
        assert len(_slugify(long)) <= 80

    def test_empty_returns_note(self):
        assert _slugify("") == "note"


@pytest.fixture()
def synthesis(tmp_path) -> Synthesis:
    p = tmp_path / "source.md"
    p.write_text("source", encoding="utf-8")
    return Synthesis(
        goal="Explain MQTT over TLS",
        markdown_body="## Goal\nExplain MQTT over TLS\n",
        source_paths=[p],
    )


class TestWriteSynthesisNote:
    def test_dry_run_does_not_write(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir, dry_run=True)
        assert result.created is False
        assert not any(out_dir.rglob("*.md")) if out_dir.exists() else True

    def test_writes_file(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir)
        assert result.created is True
        assert result.path.exists()

    def test_frontmatter_present(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir)
        content = result.path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "obsidiangpt/generated" in content
        assert "title:" in content

    def test_ai_disclaimer_callout(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir)
        content = result.path.read_text(encoding="utf-8")
        assert "[!note]" in content
        assert "AI-generated" in content

    def test_related_notes_section(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir)
        content = result.path.read_text(encoding="utf-8")
        assert "## Related Notes" in content
        assert "[[source]]" in content

    def test_slug_in_filename(self, synthesis, tmp_path):
        out_dir = tmp_path / "out"
        result = write_synthesis_note(synthesis, out_dir)
        assert "explain-mqtt-over-tls" in result.path.name

    def test_creates_output_dir(self, synthesis, tmp_path):
        out_dir = tmp_path / "deep" / "nested" / "dir"
        write_synthesis_note(synthesis, out_dir)
        assert out_dir.is_dir()
