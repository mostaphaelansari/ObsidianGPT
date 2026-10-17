"""Tests for the linker module."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsidiangpt.linker import extract_wikilinks, linkify_mentions, list_note_titles


@pytest.fixture()
def mini_vault(tmp_path):
    (tmp_path / "MQTT Broker.md").write_text("# MQTT Broker", encoding="utf-8")
    (tmp_path / "TLS Security.md").write_text("# TLS Security", encoding="utf-8")
    (tmp_path / "PostgreSQL.md").write_text("# PostgreSQL", encoding="utf-8")
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    (obsidian / "config.md").write_text("obsidian config", encoding="utf-8")
    return tmp_path


class TestListNoteTitles:
    def test_finds_all_notes(self, mini_vault):
        titles = list_note_titles(mini_vault)
        assert "MQTT Broker" in titles
        assert "TLS Security" in titles
        assert "PostgreSQL" in titles

    def test_excludes_obsidian_dir(self, mini_vault):
        titles = list_note_titles(mini_vault)
        assert "config" not in titles

    def test_empty_vault(self, tmp_path):
        assert list_note_titles(tmp_path) == set()


class TestLinkifyMentions:
    def test_wraps_known_title(self):
        text = "Discussing MQTT Broker setup."
        result = linkify_mentions(text, {"MQTT Broker"})
        assert "[[MQTT Broker]]" in result

    def test_case_insensitive_match(self):
        text = "Use mqtt broker for routing."
        result = linkify_mentions(text, {"MQTT Broker"})
        assert "[[" in result

    def test_does_not_double_link(self):
        text = "See [[MQTT Broker]] for details."
        result = linkify_mentions(text, {"MQTT Broker"})
        # Should not become [[ [[MQTT Broker]] ]]
        assert result.count("[[") == result.count("]]") == 1

    def test_longer_title_preferred(self):
        text = "Use MQTT Broker for messaging."
        result = linkify_mentions(text, {"MQTT", "MQTT Broker"})
        assert "[[MQTT Broker]]" in result

    def test_empty_known_titles_returns_unchanged(self):
        text = "Some text."
        assert linkify_mentions(text, set()) == text

    def test_skips_inline_code(self):
        text = "- `C:/Users/me/Vault/notes/Python.md`"
        assert linkify_mentions(text, {"C", "Vault", "Python"}) == text

    def test_skips_fenced_code(self):
        text = "Python here.\n```python\nimport Python\n```"
        result = linkify_mentions(text, {"Python"})
        assert result == "[[Python]] here.\n```python\nimport Python\n```"

    def test_skips_longer_fence_containing_shorter_one(self):
        text = "````text\n```\ncode\n```\n### Python\n````\nPython"
        result = linkify_mentions(text, {"Python"})
        assert result == "````text\n```\ncode\n```\n### Python\n````\n[[Python]]"

    def test_no_link_nested_inside_another(self):
        result = linkify_mentions("Linear Regression basics", {"Linear Regression", "Regression"})
        assert result == "[[Linear Regression]] basics"


class TestExtractWikilinks:
    def test_extracts_links(self):
        md = "See [[MQTT Broker]] and [[TLS Security]]."
        assert extract_wikilinks(md) == ["MQTT Broker", "TLS Security"]

    def test_empty_text(self):
        assert extract_wikilinks("") == []

    def test_no_links(self):
        assert extract_wikilinks("No links here.") == []
