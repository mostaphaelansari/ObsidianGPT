"""End-to-end pipeline tests (dry-run, no API key needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsidiangpt.config import RunConfig
from obsidiangpt.pipeline import run_goal
from obsidiangpt.planner import plan
from obsidiangpt.models import Plan, SubTask


@pytest.fixture()
def vault(tmp_path):
    """Tiny vault with two notes."""
    (tmp_path / "mqtt.md").write_text(
        "# MQTT\nMQTT is a lightweight publish/subscribe protocol. "
        "A broker routes messages between publishers and subscribers.",
        encoding="utf-8",
    )
    (tmp_path / "tls.md").write_text(
        "# TLS\nTLS (Transport Layer Security) encrypts TCP connections. "
        "MQTT can run over TLS on port 8883.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def cfg(vault, tmp_path):
    return RunConfig(
        vault_root=vault,
        output_dir=tmp_path / "out",
        dry_run=True,
        top_k_chunks=5,
    )


class TestRunGoal:
    def test_dry_run_does_not_write_file(self, mocker, cfg):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        plan_obj, chunks, synthesis, result = run_goal(cfg, "Explain MQTT")
        assert result.created is False
        assert not result.path.exists()

    def test_returns_plan_with_subtasks(self, mocker, cfg):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        plan_obj, _, _, _ = run_goal(cfg, "Explain MQTT over TLS")
        assert len(plan_obj.subtasks) >= 1

    def test_synthesis_body_not_empty(self, mocker, cfg):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        _, _, synthesis, _ = run_goal(cfg, "Explain MQTT over TLS")
        assert synthesis.markdown_body.strip()

    def test_llm_response_used_in_synthesis(self, mocker, cfg):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="1. What is MQTT?\n2. How does TLS work?")
        # Use a phrase that won't be touched by linkify_mentions (no vault-title words)
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="Pub/sub is awesome. Encryption is secure.")
        _, _, synthesis, _ = run_goal(cfg, "Explain MQTT over TLS")
        assert "is awesome" in synthesis.markdown_body

    def test_run_goal_writes_file_when_not_dry_run(self, mocker, cfg, tmp_path):
        mocker.patch("obsidiangpt.planner.call_llm", return_value="")
        mocker.patch("obsidiangpt.reasoner.call_llm", return_value="")
        cfg.dry_run = False
        p, chunks, s, res = run_goal(cfg, "My super complex goal")
        assert res.created is True
        assert res.path.exists()
