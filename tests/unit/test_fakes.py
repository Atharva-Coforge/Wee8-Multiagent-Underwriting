"""Scripted LLM adapter used in place of Ollama."""

import json
import sys
from pathlib import Path

import pytest

# Pytest's importlib mode does not put the repository root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fakes import FakeLLMAdapter
from underwriting.llm_adapter import LLMAdapter, LLMResponse, LLMTimeoutError


def use(llm: LLMAdapter) -> LLMResponse:
    return llm.complete(system="system", user="user")


def test_dict_reply_is_json_with_configured_token_counts():
    adapter = FakeLLMAdapter(
        [{"decision": "approve"}],
        prompt_tokens=17,
        completion_tokens=3,
    )

    result = adapter.complete(system="recommend", user="low score")

    assert json.loads(result.text) == {"decision": "approve"}
    assert result.prompt_tokens == 17
    assert result.completion_tokens == 3

    defaults = FakeLLMAdapter([{"decision": "approve"}])
    default_result = defaults.complete(system="recommend", user="low score")
    assert default_result.prompt_tokens == 100
    assert default_result.completion_tokens == 50


def test_string_reply_is_returned_unchanged():
    raw = '{"score": 72,'
    adapter = FakeLLMAdapter([raw])

    result = adapter.complete(system="score", user="case")

    assert result.text == raw
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.text)


def test_replies_are_returned_in_order():
    adapter = FakeLLMAdapter([{"n": 1}, "not-json", {"n": 3}])

    first = adapter.complete(system="s", user="u")
    second = adapter.complete(system="s", user="u")
    third = adapter.complete(system="s", user="u")

    assert json.loads(first.text) == {"n": 1}
    assert second.text == "not-json"
    assert json.loads(third.text) == {"n": 3}


def test_call_log_records_system_user_and_index():
    adapter = FakeLLMAdapter([{"ok": 1}, {"ok": 2}])

    adapter.complete(system="intake", user="paragraph one")
    adapter.complete(system="enrich", user="paragraph two")

    assert adapter.call_count == 2
    assert [(call.system, call.user, call.index) for call in adapter.calls] == [
        ("intake", "paragraph one", 1),
        ("enrich", "paragraph two", 2),
    ]


def test_bad_json_then_returns_invalid_then_valid():
    reply = {"score": 72, "tier": "moderate"}
    adapter = FakeLLMAdapter.bad_json_then(reply)

    first = adapter.complete(system="repair", user="broken")
    second = adapter.complete(system="repair", user="broken")

    with pytest.raises(json.JSONDecodeError):
        json.loads(first.text)
    assert json.loads(second.text) == reply


def test_timeouts_then_raises_then_returns_reply():
    reply = {"score": 40}
    adapter = FakeLLMAdapter.timeouts_then(2, reply)

    with pytest.raises(LLMTimeoutError):
        adapter.complete(system="risk", user="attempt-1")
    with pytest.raises(LLMTimeoutError):
        adapter.complete(system="risk", user="attempt-2")
    result = adapter.complete(system="risk", user="attempt-3")

    assert json.loads(result.text) == reply
    assert adapter.call_count == 3
    assert [(call.system, call.user, call.index) for call in adapter.calls] == [
        ("risk", "attempt-1", 1),
        ("risk", "attempt-2", 2),
        ("risk", "attempt-3", 3),
    ]


def test_timeouts_then_without_reply_exhausts_the_script():
    adapter = FakeLLMAdapter.timeouts_then(3)

    for _ in range(3):
        with pytest.raises(LLMTimeoutError):
            adapter.complete(system="risk", user="case")

    with pytest.raises(AssertionError, match="scripted 3 calls; another call was made"):
        adapter.complete(system="risk", user="again")


def test_fake_satisfies_llm_adapter_protocol():
    adapter = FakeLLMAdapter([{"ok": True}])

    result = use(adapter)

    assert json.loads(result.text) == {"ok": True}
