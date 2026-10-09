"""JSON completion helper: parse, one repair, no retry on timeout."""

import json

import pytest
from pydantic import BaseModel

from tests.fakes import FakeLLMAdapter
from underwriting.agents._llm_json import AgentParseError, call_json, usage_from
from underwriting.llm_adapter import LLMResponse, LLMTimeoutError
from underwriting.models import TokenUsage


class _Score(BaseModel):
    score: int


def test_valid_json_on_the_first_try_makes_one_call():
    llm = FakeLLMAdapter([{"score": 72}])

    parsed, responses = call_json(
        llm,
        agent="risk",
        system="score the case",
        user="bands",
        schema=_Score,
    )

    assert parsed.score == 72
    assert llm.call_count == 1
    assert len(responses) == 1
    assert json.loads(responses[0].text) == {"score": 72}


def test_bad_json_is_repaired_with_the_validation_error():
    bad = '{"score": 72,'
    with pytest.raises(json.JSONDecodeError) as decode_error:
        json.loads(bad)
    llm = FakeLLMAdapter.bad_json_then({"score": 72})

    parsed, responses = call_json(
        llm,
        agent="risk",
        system="score the case",
        user="bands",
        schema=_Score,
    )

    assert parsed.score == 72
    assert llm.call_count == 2
    assert len(responses) == 2
    assert llm.calls[1].system == "score the case"
    assert str(decode_error.value) in llm.calls[1].user
    assert bad in llm.calls[1].user
    assert "Return only corrected JSON." in llm.calls[1].user


def test_two_invalid_replies_raise_after_exactly_two_calls():
    second = "still not json"
    llm = FakeLLMAdapter(['{"score": 72,', second])

    with pytest.raises(AgentParseError) as caught:
        call_json(
            llm,
            agent="intake",
            system="extract",
            user="paragraph",
            schema=_Score,
        )

    with pytest.raises(json.JSONDecodeError) as decode_error:
        json.loads(second)
    assert llm.call_count == 2
    assert caught.value.agent == "intake"
    assert caught.value.validation_error == str(decode_error.value)
    assert caught.value.llm_calls == 2
    assert caught.value.usage == TokenUsage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
    )


def test_usage_from_sums_two_responses():
    usage = usage_from(
        [
            LLMResponse(text="{}", prompt_tokens=10, completion_tokens=4),
            LLMResponse(text="{}", prompt_tokens=30, completion_tokens=6),
        ]
    )

    assert usage == TokenUsage(
        prompt_tokens=40,
        completion_tokens=10,
        total_tokens=50,
    )


def test_json_fence_parses():
    llm = FakeLLMAdapter(['Here is the object:\n```json\n{"score": 15}\n```\n'])

    parsed, responses = call_json(
        llm,
        agent="risk",
        system="score",
        user="case",
        schema=_Score,
    )

    assert parsed.score == 15
    assert llm.call_count == 1
    assert len(responses) == 1


def test_json_with_surrounding_prose_parses():
    llm = FakeLLMAdapter(['Result follows {"score": 3} thanks'])

    parsed, _responses = call_json(
        llm,
        agent="risk",
        system="score",
        user="case",
        schema=_Score,
    )

    assert parsed.score == 3
    assert llm.call_count == 1


def test_schema_mismatch_is_repaired_once():
    llm = FakeLLMAdapter([{"score": "high"}, {"score": 9}])

    parsed, responses = call_json(
        llm,
        agent="risk",
        system="score",
        user="case",
        schema=_Score,
    )

    assert parsed.score == 9
    assert llm.call_count == 2
    assert len(responses) == 2
    assert "score" in llm.calls[1].user


def test_timeout_raises_without_a_repair_call():
    llm = FakeLLMAdapter.timeouts_then(1)

    with pytest.raises(LLMTimeoutError):
        call_json(
            llm,
            agent="risk",
            system="score",
            user="case",
            schema=_Score,
        )

    assert llm.call_count == 1
