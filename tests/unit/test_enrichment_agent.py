"""Enrichment agent against the JSON databases and a scripted fake."""

import json
from pathlib import Path

import pytest

from tests.fakes import FakeLLMAdapter
from underwriting.agents.enrichment_agent import enrichment_agent
from underwriting.agents.errors import AgentEscalation
from underwriting.models import IntakeRecord, TokenUsage

_LOW_REPLY = {
    "claims_band": "low",
    "violation_severity": "low",
    "vehicle_risk_band": "low",
    "prior_insurance_band": "low",
    "discrepancy_flags": [],
    "summary": "One not-at-fault glass claim and no violations.",
}


def _intake(**overrides: object) -> IntakeRecord:
    fields: dict[str, object] = {
        "case_id": "maria_ortiz",
        "raw_text": "RAW-TEXT-MARKER maria-ortiz",
        "full_name": "Maria Elena Ortiz",
        "date_of_birth": "1984-05-12",
        "vehicle_year": 2021,
        "vehicle_make": "Honda",
        "vehicle_model": "CR-V",
        "coverage": "100/300/100",
        "occupation": "registered nurse",
        "missing_fields": [],
        "decision": "proceed",
        "reason": "all required fields are present",
        "applicant_claims": "insured continuously; no accidents or tickets",
    }
    fields.update(overrides)
    return IntakeRecord(**fields)  # type: ignore[arg-type]


def _assert_raw_text_stays_out(llm: FakeLLMAdapter, marker: str) -> None:
    assert llm.call_count >= 1
    for call in llm.calls:
        assert marker not in call.user
        assert marker not in call.system
        assert "raw_text" not in call.user


def test_maria_ortiz_full_match_is_a_low_band():
    marker = "RAW-TEXT-MARKER maria-ortiz"
    intake = _intake(raw_text=marker)
    llm = FakeLLMAdapter([_LOW_REPLY])

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    assert case.lookup.match_status == "full_match"
    assert case.lookup.person_id == "P-1001"
    assert case.intake == intake
    assert case.data_gaps == []
    assert case.verified is True
    assert case.claims_band == "low"
    assert case.violation_severity == "low"
    assert case.vehicle_risk_band == "low"
    assert case.discrepancy_flags == []
    assert case.summary == _LOW_REPLY["summary"]
    assert llm.call_count == 1
    assert result.llm_calls == 1
    assert result.usage == TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    user = llm.calls[0].user
    assert "glass" in user
    assert "Windshield" in user
    _assert_raw_text_stays_out(llm, marker)
    system = llm.calls[0].system
    assert "Database rows are verified" in system
    assert "nothing at fault" in system
    assert "two or more at-fault accidents, or any bodily injury" in system


def test_priya_shah_keeps_the_scripted_discrepancy_flag():
    marker = "RAW-TEXT-MARKER priya-shah"
    claim = "clean record, no accidents and no tickets"
    flag = (
        "applicant claims no accidents; driving history has 1 at-fault "
        "accident on 2025-11-02"
    )
    intake = _intake(
        case_id="clean_claim_mismatch",
        raw_text=marker,
        full_name="Priya N. Shah",
        date_of_birth="1994-07-19",
        vehicle_year=2018,
        vehicle_make="Subaru",
        vehicle_model="Outback",
        occupation="teacher",
        applicant_claims=claim,
    )
    reply = {
        "claims_band": "moderate",
        "violation_severity": "moderate",
        "vehicle_risk_band": "low",
        "prior_insurance_band": "low",
        "discrepancy_flags": [flag],
        "summary": "The clean-record claim conflicts with an at-fault accident.",
    }
    llm = FakeLLMAdapter([reply])

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    user = llm.calls[0].user
    assert claim in user
    assert "2025-11-02" in user
    assert "Ran a red light and struck a turning vehicle." in user
    assert case.discrepancy_flags == [flag]
    assert case.lookup.person_id == "P-1003"
    assert case.lookup.match_status == "full_match"
    _assert_raw_text_stays_out(llm, marker)


def test_tyler_brandt_prior_insurance_band_is_severe():
    marker = "RAW-TEXT-MARKER tyler-brandt"
    intake = _intake(
        case_id="tyler_brandt",
        raw_text=marker,
        full_name="Tyler James Brandt",
        date_of_birth="2002-01-30",
        vehicle_year=2020,
        vehicle_make="Dodge",
        vehicle_model="Charger",
        coverage="50/100/50",
        occupation="warehouse associate",
        applicant_claims="prior policy was non-renewed",
    )
    reply = {
        "claims_band": "severe",
        "violation_severity": "severe",
        "vehicle_risk_band": "high",
        "prior_insurance_band": "severe",
        "discrepancy_flags": [],
        "summary": "A 50-day lapse and a non-renewal.",
    }
    llm = FakeLLMAdapter([reply])

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    assert case.lookup.match_status == "full_match"
    assert case.lookup.person_id == "P-1002"
    user = llm.calls[0].user
    assert '"lapse_days": 50' in user
    assert '"non_renewed": true' in user
    assert case.prior_insurance_band == "severe"
    _assert_raw_text_stays_out(llm, marker)


def test_robin_hale_records_the_prior_insurance_gap():
    marker = "RAW-TEXT-MARKER robin-hale"
    gap = "no prior insurance record found"
    intake = _intake(
        case_id="robin_hale",
        raw_text=marker,
        full_name="Robin Hale",
        date_of_birth="1980-01-15",
        vehicle_year=2015,
        vehicle_make="Toyota",
        vehicle_model="Camry",
        occupation=None,
        applicant_claims="",
    )
    llm = FakeLLMAdapter([_LOW_REPLY])

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    assert case.lookup.match_status == "full_match"
    assert case.lookup.person_id == "P-1007"
    assert case.data_gaps == [gap]
    assert gap in llm.calls[0].user
    _assert_raw_text_stays_out(llm, marker)


def test_jordan_washington_dob_mismatch_does_not_call_the_model():
    intake = _intake(
        case_id="dob_mismatch",
        raw_text="RAW-TEXT-MARKER jordan-washington",
        full_name="Jordan A. Washington",
        date_of_birth="1990-03-14",
        vehicle_year=2019,
        vehicle_make="Honda",
        vehicle_model="Civic",
        coverage="100/300/50",
        occupation="teacher",
        applicant_claims="",
    )
    llm = FakeLLMAdapter([])

    with pytest.raises(AgentEscalation) as caught:
        enrichment_agent(intake, llm=llm)

    error = caught.value
    assert error.status == "escalated"
    assert error.decision == "refer"
    assert error.reason == "possible identity mismatch"
    assert error.case_id == "dob_mismatch"
    assert error.llm_calls == 0
    assert error.usage == TokenUsage()
    assert error.record is not None
    assert error.record.match_status == "dob_mismatch"
    assert llm.call_count == 0


def test_samir_cole_not_found_does_not_call_the_model():
    intake = _intake(
        case_id="unknown_person",
        raw_text="RAW-TEXT-MARKER samir-cole",
        full_name="Samir Cole",
        date_of_birth="1995-06-01",
        vehicle_year=2016,
        vehicle_make="Toyota",
        vehicle_model="Corolla",
        occupation="teacher",
        applicant_claims="",
    )
    llm = FakeLLMAdapter([])

    with pytest.raises(AgentEscalation) as caught:
        enrichment_agent(intake, llm=llm)

    error = caught.value
    assert error.status == "escalated"
    assert error.decision == "refer"
    assert error.reason == "insufficient information found"
    assert error.llm_calls == 0
    assert error.record is not None
    assert error.record.match_status == "not_found"
    assert llm.call_count == 0


def test_missing_driving_history_row_is_insufficient_information(tmp_path: Path):
    _write(tmp_path / "driving_history.json", [])
    _write(
        tmp_path / "vehicles_and_drivers.json",
        [
            {
                "person_id": "P-1008",
                "full_name": "Casey Nguyen",
                "date_of_birth": "1985-09-09",
                "vehicles": [
                    {
                        "year": 2016,
                        "make": "Ford",
                        "model": "Focus",
                        "currently_insured": True,
                    }
                ],
            }
        ],
    )
    _write(
        tmp_path / "prior_insurance.json",
        [
            {
                "person_id": "P-1008",
                "full_name": "Casey Nguyen",
                "date_of_birth": "1985-09-09",
                "policies": [
                    {
                        "carrier": "State Auto",
                        "currently_insured": True,
                        "years_continuous": 3,
                        "lapse_days": 0,
                        "cancelled": False,
                        "non_renewed": False,
                    }
                ],
            }
        ],
    )
    intake = _intake(
        case_id="casey_nguyen",
        raw_text="RAW-TEXT-MARKER casey-nguyen",
        full_name="Casey Nguyen",
        date_of_birth="1985-09-09",
        vehicle_year=2016,
        vehicle_make="Ford",
        vehicle_model="Focus",
        occupation=None,
        applicant_claims="",
    )
    llm = FakeLLMAdapter([])

    with pytest.raises(AgentEscalation) as caught:
        enrichment_agent(intake, llm=llm, database_dir=tmp_path)

    error = caught.value
    assert error.status == "escalated"
    assert error.decision == "refer"
    assert error.reason == "insufficient information found"
    assert error.llm_calls == 0
    assert error.record is not None
    assert error.record.match_status == "not_found"
    assert error.record.driving_history is None
    assert llm.call_count == 0


def test_bad_json_is_repaired_through_the_agent():
    marker = "RAW-TEXT-MARKER maria-repair"
    intake = _intake(raw_text=marker)
    llm = FakeLLMAdapter.bad_json_then(_LOW_REPLY)

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    assert llm.call_count == 2
    assert result.llm_calls == 2
    assert result.usage == TokenUsage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
    )
    assert case.lookup.person_id == "P-1001"
    assert case.claims_band == "low"
    assert case.intake == intake
    _assert_raw_text_stays_out(llm, marker)


def test_data_gaps_come_from_code_not_the_model():
    marker = "RAW-TEXT-MARKER robin-gaps"
    gap = "no prior insurance record found"
    invented = "the model invented a prior-carrier gap"
    intake = _intake(
        case_id="robin_hale",
        raw_text=marker,
        full_name="Robin Hale",
        date_of_birth="1980-01-15",
        vehicle_year=2015,
        vehicle_make="Toyota",
        vehicle_model="Camry",
        occupation=None,
        applicant_claims="",
    )
    reply = {
        **_LOW_REPLY,
        "data_gaps": [invented],
    }
    llm = FakeLLMAdapter([reply])

    result = enrichment_agent(intake, llm=llm)
    case = result.output

    assert case.data_gaps == [gap]
    assert invented not in case.data_gaps
    assert llm.call_count == 1
    _assert_raw_text_stays_out(llm, marker)


def test_system_prompt_has_field_priority_and_three_examples():
    intake = _intake()
    llm = FakeLLMAdapter([_LOW_REPLY])

    enrichment_agent(intake, llm=llm)

    system = llm.calls[0].system
    assert "Field priority" in system
    assert "Example 1 (clean)" in system
    assert "Example 2 (messy)" in system
    assert "Example 3 (risky)" in system


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"records": records}))
