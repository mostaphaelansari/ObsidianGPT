"""Tests for the LLM-backed planner."""
from __future__ import annotations

from unittest.mock import patch

from obsidiangpt.planner import _parse_numbered_list, _split_goal, plan
from obsidiangpt.models import Plan


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

class TestSplitGoal:
    def test_multiline_splits_by_line(self):
        result = _split_goal("One thing\nTwo thing")
        assert len(result) == 2

    def test_single_goal_returns_three_defaults(self):
        result = _split_goal("Single short goal")
        assert len(result) == 3

    def test_empty_returns_empty(self):
        assert _split_goal("") == []

    def test_semicolon_delimiter(self):
        result = _split_goal("Part A; Part B; Part C")
        assert len(result) == 3


class TestParseNumberedList:
    def test_standard_numbered_list(self):
        text = "1. What is MQTT?\n2. How does TLS work?\n3. What is PostgreSQL?"
        result = _parse_numbered_list(text)
        assert result == [
            "What is MQTT?",
            "How does TLS work?",
            "What is PostgreSQL?",
        ]

    def test_paren_style(self):
        text = "1) First\n2) Second"
        result = _parse_numbered_list(text)
        assert result == ["First", "Second"]

    def test_ignores_non_list_lines(self):
        text = "Here is the plan:\n1. Do this\n2. Then that\nEnd."
        result = _parse_numbered_list(text)
        assert len(result) == 2

    def test_empty_returns_empty(self):
        assert _parse_numbered_list("") == []


# ---------------------------------------------------------------------------
# LLM path (mocked)
# ---------------------------------------------------------------------------

class TestPlan:
    def test_llm_path_uses_response(self, mocker):
        mock_llm = mocker.patch(
            "obsidiangpt.planner.call_llm",
            return_value="1. Sub-question A?\n2. Sub-question B?\n3. Sub-question C?",
        )
        p = plan("Design an MQTT system")
        assert isinstance(p, Plan)
        assert len(p.subtasks) == 3
        assert p.subtasks[0].question == "Sub-question A?"
        mock_llm.assert_called_once()

    def test_fallback_when_llm_returns_empty(self, mocker):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        p = plan("Single short goal")
        assert len(p.subtasks) >= 1

    def test_fallback_when_llm_unparseable(self, mocker):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="Sure, here's a plan!")
        p = plan("Some goal")
        # Should fall back to heuristic, which returns >= 1 subtasks
        assert len(p.subtasks) >= 1

    def test_model_passed_to_llm(self, mocker):
        mock_llm = mocker.patch(
            "obsidiangpt.planner.call_llm",
            return_value="1. Q one?\n2. Q two?",
        )
        plan("goal", model="gemini-2.5-flash")
        _, kwargs = mock_llm.call_args
        assert kwargs["model"] == "gemini-2.5-flash"

    def test_plan_splits_multiline(self, mocker):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        p = plan("One thing\nTwo thing")
        assert len(p.subtasks) == 2

    def test_plan_fallback(self, mocker):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        p = plan("Single short goal")
        assert len(p.subtasks) >= 1
