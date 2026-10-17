"""Tests for LLM provider selection and fallback."""
from __future__ import annotations

import pytest

from obsidiangpt import llm
from obsidiangpt.llm import LLMError, active_provider, call_llm, resolve_model

_ENV_VARS = [
    "LLM_PROVIDER",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "ANTHROPIC_MODEL", "OPENAI_MODEL", "GEMINI_MODEL", "OLLAMA_MODEL",
    "OLLAMA_HOST",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def backends(mocker):
    """Replace every backend with a mock that returns its provider name."""
    mocks = {
        name: mocker.Mock(return_value=f"from {name}")
        for name in ("anthropic", "openai", "gemini")
    }
    mocker.patch.dict(llm._CLOUD_BACKENDS, mocks)
    mocks["ollama"] = mocker.patch("obsidiangpt.llm._call_ollama", return_value="from ollama")
    return mocks


class TestActiveProvider:
    def test_no_keys_uses_ollama(self):
        assert active_provider() == "ollama"

    @pytest.mark.parametrize("key_env, provider", [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
    ])
    def test_api_key_selects_provider(self, monkeypatch, key_env, provider):
        monkeypatch.setenv(key_env, "secret")
        assert active_provider() == provider

    def test_blank_key_is_ignored(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        assert active_provider() == "ollama"

    def test_first_key_wins(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        assert active_provider() == "anthropic"

    def test_llm_provider_forces_choice(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("LLM_PROVIDER", "Ollama")
        assert active_provider() == "ollama"

    def test_unknown_llm_provider_falls_back_to_detection(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nope")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert active_provider() == "openai"


class TestResolveModel:
    def test_explicit_model_wins(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        assert resolve_model("ollama", "mistral") == "mistral"

    def test_env_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
        assert resolve_model("openai") == "gpt-x"

    def test_default_model(self):
        assert resolve_model("anthropic") == "claude-opus-5-5"
        assert resolve_model("ollama") == "qwen3:1.7b"


class TestCallLLM:
    def test_no_keys_calls_ollama(self, backends):
        assert call_llm("sys", "user") == "from ollama"
        backends["ollama"].assert_called_once_with("sys", "user", "qwen3:1.7b", "http://localhost:11434")

    def test_api_key_calls_cloud_not_ollama(self, monkeypatch, backends):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        assert call_llm("sys", "user", model="claude-sonnet-5-5") == "from anthropic"
        backends["anthropic"].assert_called_once_with("sys", "user", "claude-sonnet-5-5", "secret")
        backends["ollama"].assert_not_called()

    def test_cloud_failure_falls_back_to_ollama_default_model(self, monkeypatch, backends):
        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        backends["openai"].side_effect = LLMError("boom")
        assert call_llm("sys", "user", model="gpt-x") == "from ollama"
        # The cloud model name must not be sent to Ollama.
        assert backends["ollama"].call_args.args[2] == "qwen3:1.7b"

    def test_everything_down_returns_empty(self, backends):
        backends["ollama"].side_effect = LLMError("down")
        assert call_llm("sys", "user") == ""
