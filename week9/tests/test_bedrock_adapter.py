"""BedrockAdapter tests with a fake Converse client. No AWS calls."""

import json
from pathlib import Path

import pytest
from week9.risk_agent.bedrock_adapter import DEFAULT_MODEL_ID, BedrockAdapter

from underwriting.agents.risk_scoring_agent import risk_scoring_agent
from underwriting.llm_adapter import LLMTimeoutError
from underwriting.models import EnrichedCase
from underwriting.pipeline.retry import retry_on_timeout

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class _ReadTimeout(Exception):
    """Stands in for botocore's ReadTimeoutError."""


class _Client:
    """Fake bedrock-runtime client. Each reply is a response dict or an exception."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _response(text: str, input_tokens: int = 11, output_tokens: int = 4) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


def _adapter(client: _Client, **kwargs) -> BedrockAdapter:
    return BedrockAdapter(client=client, timeout_errors=(_ReadTimeout,), **kwargs)


def test_complete_returns_text_and_token_counts():
    client = _Client(_response('{"ok": true}', 11, 4))

    result = _adapter(client).complete(system="system", user="user")

    assert result.text == '{"ok": true}'
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4


def test_request_sends_system_and_user_at_temperature_zero():
    client = _Client(_response("{}"))

    _adapter(client, max_tokens=512).complete(system="rules", user="case")

    call = client.calls[0]
    assert call["modelId"] == DEFAULT_MODEL_ID == "google.gemma-3-27b-it"
    assert call["system"] == [{"text": "rules"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "case"}]}]
    assert call["inferenceConfig"] == {"temperature": 0, "maxTokens": 512}


def test_system_in_user_puts_the_system_prompt_first_in_the_user_message():
    client = _Client(_response("{}"))

    _adapter(client, system_in_user=True).complete(system="rules", user="case")

    call = client.calls[0]
    assert "system" not in call
    assert call["messages"][0]["content"] == [{"text": "rules\n\ncase"}]


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "google.gemma-3-12b-it")
    monkeypatch.setenv("BEDROCK_SYSTEM_IN_USER", "1")
    client = _Client(_response("{}"))

    _adapter(client).complete(system="rules", user="case")

    call = client.calls[0]
    assert call["modelId"] == "google.gemma-3-12b-it"
    assert "system" not in call


def test_text_blocks_are_joined_and_missing_usage_counts_zero():
    client = _Client(
        {"output": {"message": {"content": [{"text": '{"a": '}, {"text": "1}"}]}}}
    )

    result = _adapter(client).complete(system="s", user="u")

    assert result.text == '{"a": 1}'
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_timeout_raises_llm_timeout_error():
    client = _Client(_ReadTimeout("Read timeout on endpoint URL"))

    with pytest.raises(LLMTimeoutError, match="Read timeout"):
        _adapter(client).complete(system="s", user="u")


def test_other_errors_are_not_turned_into_timeouts():
    client = _Client(RuntimeError("AccessDeniedException"))

    with pytest.raises(RuntimeError, match="AccessDenied"):
        _adapter(client).complete(system="s", user="u")


def test_risk_scoring_agent_runs_on_bedrock_after_one_timeout():
    """The week 8 agent and retry work unchanged on the Bedrock adapter."""
    enriched = EnrichedCase.model_validate_json(
        (_FIXTURES / "maria_ortiz.json").read_text()
    )
    reply = {
        "score": 15,
        "factors": [
            {"name": "claims_band low", "direction": "decrease", "weight": "low"},
        ],
    }
    client = _Client(_ReadTimeout("timed out"), _response(json.dumps(reply), 1200, 90))
    adapter = _adapter(client)

    attempts = list(
        retry_on_timeout(
            lambda _attempt: risk_scoring_agent(enriched, llm=adapter),
            sleep=lambda _seconds: None,
        )
    )

    assert [attempt.error is not None for attempt in attempts] == [True, False]
    result = attempts[-1].value
    assert result is not None
    assert result.output.score == 15
    assert result.output.tier == "low"
    assert result.usage.total_tokens == 1290
    assert len(client.calls) == 2
    sent = client.calls[1]["messages"][0]["content"][0]["text"]
    assert '"claims_band": "low"' in sent
    assert "Maria" not in sent
