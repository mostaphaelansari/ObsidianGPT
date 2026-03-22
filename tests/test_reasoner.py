"""Tests for the LLM-backed reasoner."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsidiangpt.models import Plan, RetrievedChunk, SubTask
from obsidiangpt.reasoner import _build_context, _template_synthesis, reason


@pytest.fixture()
def simple_plan():
    return Plan(
        goal="Explain MQTT",
        subtasks=[
            SubTask(id="a1b2c3d4", question="What is MQTT?"),
            SubTask(id="e5f6a7b8", question="How does TLS secure MQTT?"),
        ],
    )


@pytest.fixture()
def sample_chunks(tmp_path):
    p1 = tmp_path / "mqtt.md"
    p1.write_text("MQTT is a lightweight pub/sub protocol.", encoding="utf-8")
    p2 = tmp_path / "tls.md"
    p2.write_text("TLS encrypts TCP connections.", encoding="utf-8")
    return [
        RetrievedChunk(path=p1, excerpt="MQTT is a lightweight pub/sub protocol.", score=0.9),
        RetrievedChunk(path=p2, excerpt="TLS encrypts TCP connections.", score=0.7),
    ]


class TestBuildContext:
    def test_includes_goal(self, simple_plan, sample_chunks):
        ctx = _build_context(simple_plan, sample_chunks)
        assert "Explain MQTT" in ctx

    def test_includes_sub_questions(self, simple_plan, sample_chunks):
        ctx = _build_context(simple_plan, sample_chunks)
        assert "What is MQTT?" in ctx
        assert "How does TLS secure MQTT?" in ctx

    def test_includes_excerpt(self, simple_plan, sample_chunks):
        ctx = _build_context(simple_plan, sample_chunks)
        assert "lightweight pub/sub" in ctx

    def test_no_chunks_message(self, simple_plan):
        ctx = _build_context(simple_plan, [])
        assert "No relevant notes" in ctx


class TestTemplateSynthesis:
    def test_contains_goal_section(self, simple_plan, sample_chunks):
        body = _template_synthesis(simple_plan, sample_chunks)
        assert "## Goal" in body

    def test_contains_evidence_section(self, simple_plan, sample_chunks):
        body = _template_synthesis(simple_plan, sample_chunks)
        assert "## Evidence from vault" in body

    def test_contains_sources_section(self, simple_plan, sample_chunks):
        body = _template_synthesis(simple_plan, sample_chunks)
        assert "## Sources" in body

    def test_no_chunks_shows_offline_note(self, simple_plan):
        body = _template_synthesis(simple_plan, [])
        assert "No matching notes" in body


class TestReason:
    def test_llm_path_uses_response(self, mocker, simple_plan, sample_chunks):
        expected = "### Overview\nMQTT is great.\n## Contradictions & Gaps\nNone.\n## Sources\n- `mqtt.md`"
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value=expected)
        synthesis = reason(simple_plan, sample_chunks)
        assert "MQTT is great" in synthesis.markdown_body

    def test_fallback_when_llm_empty(self, mocker, simple_plan, sample_chunks):
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        synthesis = reason(simple_plan, sample_chunks)
        assert "## Goal" in synthesis.markdown_body

    def test_source_paths_populated(self, mocker, simple_plan, sample_chunks):
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="Some synthesis.")
        synthesis = reason(simple_plan, sample_chunks)
        assert len(synthesis.source_paths) == 2

    def test_model_forwarded_to_llm(self, mocker, simple_plan, sample_chunks):
        mock_llm = mocker.patch("obsidiangpt.reasoner.call_llm", return_value="answer")
        reason(simple_plan, sample_chunks, model="gemini-2.5-pro")
        _, kwargs = mock_llm.call_args
        assert kwargs["model"] == "gemini-2.5-pro"

    def test_no_chunks_still_produces_synthesis(self, mocker, simple_plan):
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        synthesis = reason(simple_plan, [])
        assert synthesis.goal == "Explain MQTT"
        assert synthesis.markdown_body
