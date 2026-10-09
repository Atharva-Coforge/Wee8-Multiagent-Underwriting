"""Model contracts for intake, database rows, and pipeline outcomes."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from underwriting.models import (
    DrivingHistoryRecord,
    IntakeRecord,
    PipelineResult,
    PriorInsuranceRecord,
    Recommendation,
    VehicleRecord,
)

DATABASES = Path(__file__).resolve().parents[2] / "data" / "databases"


def _records(filename: str) -> list[dict]:
    payload = json.loads((DATABASES / filename).read_text())
    return payload["records"]


def test_driving_history_records_parse():
    records = [DrivingHistoryRecord.model_validate(row) for row in _records("driving_history.json")]

    assert len(records) == 7
    maria = records[0]
    assert maria.person_id == "P-1001"
    assert maria.claims[0].type == "glass"
    assert maria.claims[0].description.startswith("Windshield")
    tyler = records[1]
    assert len(tyler.accidents) == 2
    assert len(tyler.violations) == 3
    assert tyler.claims[0].type == "collision"
    assert tyler.claims[0].bodily_injury is True


def test_vehicle_records_parse():
    records = [VehicleRecord.model_validate(row) for row in _records("vehicles_and_drivers.json")]

    assert len(records) == 7
    assert records[0].vehicles[0].year == 2021
    assert records[0].vehicles[0].model == "CR-V"
    assert records[-1].person_id == "P-1007"


def test_prior_insurance_records_parse():
    records = [PriorInsuranceRecord.model_validate(row) for row in _records("prior_insurance.json")]

    assert len(records) == 6
    assert records[0].policies[0].carrier == "Buckeye Mutual"
    tyler = records[1]
    assert tyler.policies[0].lapse_days == 50
    assert tyler.policies[0].non_renewed is True
    assert all(row.person_id != "P-1007" for row in records)


def test_intake_record_allows_missing_fields():
    record = IntakeRecord(
        case_id="incomplete",
        raw_text="My name is Kevin Osei. I will call my date of birth in this afternoon.",
        full_name="Kevin Osei",
        vehicle_year=2019,
        vehicle_make="Ford",
        vehicle_model="F-150",
        coverage="100/300/100",
        occupation="electrician",
        missing_fields=["date_of_birth"],
        decision="reject",
        reason="required information date of birth is missing",
        model_decision="proceed",
        override=True,
    )

    assert record.date_of_birth is None
    assert record.missing_fields == ["date_of_birth"]
    assert record.decision == "reject"
    assert record.override is True
    assert record.applicant_claims == ""


def test_intake_record_allows_missing_vehicle_model():
    record = IntakeRecord(
        case_id="missing_vehicle_model",
        raw_text="Lena Park, born 1992-04-04, needs coverage. Model not stated.",
        full_name="Lena Park",
        date_of_birth="1992-04-04",
        vehicle_year=2017,
        vehicle_make="Toyota",
        coverage="100/300/100",
        missing_fields=["vehicle_model"],
        decision="escalate",
        reason="required information vehicle model is missing",
    )

    assert record.vehicle_model is None
    assert record.decision == "escalate"
    assert record.model_decision is None
    assert record.override is False


def test_literals_reject_bad_values():
    with pytest.raises(ValidationError):
        PipelineResult(
            status="completed",
            decision="maybe",
            rationale="not a decision",
            case_id="maria_ortiz",
        )

    with pytest.raises(ValidationError):
        PipelineResult(
            status="referred_incomplete",
            decision="refer",
            rationale="old status",
            case_id="incomplete",
        )

    with pytest.raises(ValidationError):
        IntakeRecord(
            case_id="incomplete",
            raw_text="text",
            decision="approve",
            reason="intake cannot approve",
        )


def test_escalated_and_rejected_results_need_no_recommendation():
    rejected = PipelineResult(
        status="rejected",
        decision="reject",
        rationale="required information date of birth is missing",
        case_id="incomplete",
    )
    escalated = PipelineResult(
        status="escalated",
        decision="refer",
        rationale="required information vehicle model is missing",
        case_id="missing_vehicle_model",
    )

    assert rejected.trace is None
    assert escalated.trace is None
    assert not hasattr(rejected, "recommendation")
    fields = set(Recommendation.model_fields)
    assert "assessment" in fields
    assert "override" in fields
    assert "model_decision" in fields
