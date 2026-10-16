"""Entrypoint tests: fixture in, one JSON line out, using the fake adapter."""

import json
from pathlib import Path

import pytest
from tests.fakes import FakeLLMAdapter
from week9.risk_agent.run_risk_agent import main

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_MARIA = str(_FIXTURES / "maria_ortiz.json")
_TYLER = str(_FIXTURES / "tyler_brandt.json")

_LOW = {
    "score": 15,
    "factors": [{"name": "claims_band low", "direction": "decrease", "weight": "low"}],
}
_SEVERE = {
    "score": 92,
    "factors": [
        {"name": "claims_band severe", "direction": "increase", "weight": "high"}
    ],
}


def _lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_scores_a_fixture_and_prints_one_json_line(capsys):
    llm = FakeLLMAdapter([_LOW])

    code = main([_MARIA], llm=llm, sleep=lambda _s: None)

    assert code == 0
    [record] = _lines(capsys)
    assert record["case_id"] == "maria_ortiz"
    assert record["status"] == "scored"
    assert record["score"] == 15
    assert record["tier"] == "low"
    assert record["factors"][0]["name"] == "claims_band low"
    assert record["attempts"] == 1
    assert record["total_tokens"] == 150
    assert record["llm_calls"] == 1


def test_scores_several_fixtures_in_order(capsys):
    llm = FakeLLMAdapter([_LOW, _SEVERE])

    code = main([_MARIA, _TYLER], llm=llm, sleep=lambda _s: None)

    assert code == 0
    records = _lines(capsys)
    assert [r["case_id"] for r in records] == ["maria_ortiz", "tyler_brandt"]
    assert [r["tier"] for r in records] == ["low", "severe"]


def test_recovers_after_two_timeouts(capsys):
    llm = FakeLLMAdapter.timeouts_then(2, _LOW)
    waits: list[float] = []

    code = main([_MARIA], llm=llm, sleep=waits.append)

    assert code == 0
    [record] = _lines(capsys)
    assert record["status"] == "scored"
    assert record["attempts"] == 3
    assert waits == [1, 2]


def test_three_timeouts_escalate_to_refer(capsys):
    llm = FakeLLMAdapter.timeouts_then(3)

    code = main([_MARIA], llm=llm, sleep=lambda _s: None)

    assert code == 1
    [record] = _lines(capsys)
    assert record["status"] == "escalated"
    assert record["decision"] == "refer"
    assert record["reason"] == (
        "risk scoring timed out after 3 attempts; sent to human review"
    )
    assert record["attempts"] == 3
    assert "score" not in record


def test_parse_failure_escalates_with_the_tokens_spent(capsys):
    llm = FakeLLMAdapter(["not json", "still not json"])

    code = main([_MARIA], llm=llm, sleep=lambda _s: None)

    assert code == 1
    [record] = _lines(capsys)
    assert record["status"] == "escalated"
    assert "risk_scoring JSON parse failed" in record["reason"]
    assert record["attempts"] == 1
    assert record["llm_calls"] == 2
    assert record["total_tokens"] == 300


def test_missing_input_file_exits_2_before_any_model_call(tmp_path, capsys):
    llm = FakeLLMAdapter([])

    code = main([_MARIA, str(tmp_path / "missing.json")], llm=llm)

    assert code == 2
    assert llm.call_count == 0
    assert "cannot read" in capsys.readouterr().err


def test_invalid_input_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    llm = FakeLLMAdapter([])

    code = main([str(bad)], llm=llm)

    assert code == 2
    assert llm.call_count == 0


def test_other_errors_are_not_caught():
    llm = FakeLLMAdapter([RuntimeError("AccessDeniedException")])

    with pytest.raises(RuntimeError, match="AccessDenied"):
        main([_MARIA], llm=llm, sleep=lambda _s: None)
