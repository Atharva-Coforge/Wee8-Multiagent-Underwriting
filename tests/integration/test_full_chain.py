"""Full chain against a scripted fake. No Ollama."""

import json
from pathlib import Path

import pytest

from tests.fakes import FakeLLMAdapter
from underwriting.llm_adapter import LLMTimeoutError
from underwriting.models import EnrichedCase, IntakeRecord, Recommendation, RiskAssessment
from underwriting.pipeline.chain import run_pipeline

_APPLICATIONS = Path(__file__).resolve().parents[2] / "data" / "applications"


def _text(filename: str) -> str:
    return (_APPLICATIONS / filename).read_text()


def _intake(
    full_name: str,
    date_of_birth: str | None,
    *,
    vehicle_year: int = 2021,
    vehicle_make: str = "Honda",
    vehicle_model: str = "CR-V",
    coverage: str = "100/300/100",
    occupation: str = "registered nurse",
    applicant_claims: str = "",
    decision: str = "proceed",
    reason: str = "all required fields are present",
) -> dict[str, object]:
    return {
        "full_name": full_name,
        "date_of_birth": date_of_birth,
        "vehicle_year": vehicle_year,
        "vehicle_make": vehicle_make,
        "vehicle_model": vehicle_model,
        "coverage": coverage,
        "occupation": occupation,
        "applicant_claims": applicant_claims,
        "missing_fields": [],
        "decision": decision,
        "reason": reason,
    }


def _enrichment(
    *,
    claims_band: str = "low",
    violation_severity: str = "low",
    vehicle_risk_band: str = "low",
    prior_insurance_band: str = "low",
    discrepancy_flags: list[str] | None = None,
    summary: str = "No at-fault accidents and no violations.",
) -> dict[str, object]:
    return {
        "claims_band": claims_band,
        "violation_severity": violation_severity,
        "vehicle_risk_band": vehicle_risk_band,
        "prior_insurance_band": prior_insurance_band,
        "discrepancy_flags": [] if discrepancy_flags is None else discrepancy_flags,
        "summary": summary,
    }


def _risk(score: int = 18) -> dict[str, object]:
    return {
        "score": score,
        "factors": [
            {"name": "claims_band low", "direction": "decrease", "weight": "low"},
            {"name": "violation_severity low", "direction": "decrease", "weight": "low"},
        ],
    }


def _recommendation(
    decision: str = "approve",
    rationale: str = "A score of 18 is a low tier.",
    conditions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "decision": decision,
        "rationale": rationale,
        "conditions": [] if conditions is None else conditions,
    }


def test_maria_completes_with_approve(tmp_path: Path):
    llm = FakeLLMAdapter(
        [
            _intake(
                "Maria Elena Ortiz",
                "1984-05-12",
                applicant_claims="insured continuously; no accidents or tickets",
            ),
            _enrichment(),
            _risk(18),
            _recommendation(),
        ]
    )

    result = run_pipeline(
        _text("maria_ortiz.txt"),
        case_id="maria_ortiz",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "completed"
    assert result.decision == "approve"
    assert result.case_id == "maria_ortiz"
    assert result.rationale == "A score of 18 is a low tier."
    assert [step.agent for step in result.agents] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]
    assert all(step.llm_calls == 1 and not step.stopped for step in result.agents)
    intake, enriched, assessment, recommendation = (step.output for step in result.agents)
    assert isinstance(intake, IntakeRecord)
    assert isinstance(enriched, EnrichedCase)
    assert isinstance(assessment, RiskAssessment)
    assert isinstance(recommendation, Recommendation)
    assert enriched.intake is intake
    assert assessment.enriched is enriched
    assert recommendation.assessment is assessment
    assert recommendation.decision == "approve"
    assert recommendation.override is False
    assert llm.call_count == 4
    spans = [
        json.loads(line)
        for line in (tmp_path / "maria_ortiz.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [span["agent"] for span in spans] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]


def test_priya_completes_with_refer_on_a_discrepancy(tmp_path: Path):
    flag = (
        "applicant claims no accidents; driving history has 1 at-fault "
        "accident on 2025-11-02"
    )
    llm = FakeLLMAdapter(
        [
            _intake(
                "Priya N. Shah",
                "1994-07-19",
                vehicle_year=2018,
                vehicle_make="Subaru",
                vehicle_model="Outback",
                occupation="teacher",
                applicant_claims="clean record, no accidents and no tickets",
            ),
            _enrichment(
                claims_band="moderate",
                violation_severity="high",
                discrepancy_flags=[flag],
                summary="A clean-record claim conflicts with an at-fault accident.",
            ),
            _risk(55),
            _recommendation("approve", "The bands look acceptable."),
            {"rationale": "A discrepancy flag blocks approval and the case is referred."},
        ]
    )

    result = run_pipeline(
        _text("clean_claim_mismatch.txt"),
        case_id="clean_claim_mismatch",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "completed"
    assert result.decision == "refer"
    assert result.rationale == "A discrepancy flag blocks approval and the case is referred."
    enriched = result.agents[1].output
    recommendation = result.agents[3].output
    assert isinstance(enriched, EnrichedCase)
    assert isinstance(recommendation, Recommendation)
    assert enriched.discrepancy_flags == [flag]
    assert recommendation.decision == "refer"
    assert recommendation.override is True
    assert recommendation.model_decision == "approve"
    assert recommendation.conditions == []
    assert result.agents[3].llm_calls == 2
    assert llm.call_count == 5


def test_samir_cole_stops_escalated_for_insufficient_information(tmp_path: Path):
    llm = FakeLLMAdapter(
        [
            _intake(
                "Samir Cole",
                "1995-06-01",
                vehicle_year=2016,
                vehicle_make="Toyota",
                vehicle_model="Corolla",
                occupation="teacher",
            )
        ]
    )

    result = run_pipeline(
        _text("unknown_person.txt"),
        case_id="unknown_person",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "escalated"
    assert result.decision == "refer"
    assert result.rationale == "insufficient information found"
    assert result.case_id == "unknown_person"
    assert [step.agent for step in result.agents] == ["intake", "enrichment"]
    assert result.agents[0].stopped is False
    assert result.agents[1].stopped is True
    assert result.agents[1].llm_calls == 0
    assert llm.call_count == 1


def test_kevin_osei_stops_rejected(tmp_path: Path):
    llm = FakeLLMAdapter(
        [
            _intake(
                "Kevin Osei",
                None,
                vehicle_year=2019,
                vehicle_make="Ford",
                vehicle_model="F-150",
                occupation="electrician",
                decision="proceed",
                reason="the applicant will call the date of birth in later",
            )
        ]
    )

    result = run_pipeline(
        _text("incomplete.txt"),
        case_id="incomplete",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "rejected"
    assert result.decision == "reject"
    assert result.rationale == "required information date of birth is missing"
    assert result.case_id == "incomplete"
    assert [step.agent for step in result.agents] == ["intake"]
    assert result.agents[0].stopped is True
    intake = result.agents[0].output
    assert isinstance(intake, IntakeRecord)
    assert intake.decision == "reject"
    assert llm.call_count == 1


def test_parse_failure_escalates_without_raising(tmp_path: Path):
    llm = FakeLLMAdapter(["not json", "still not json"])

    result = run_pipeline(
        "not an application",
        case_id="bad_json",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "escalated"
    assert result.decision == "refer"
    assert result.case_id == "bad_json"
    assert result.rationale.startswith("intake JSON parse failed:")
    assert result.agents[0].agent == "intake"
    assert result.agents[0].stopped is True
    assert result.agents[0].llm_calls == 2


def test_timeout_propagates(tmp_path: Path):
    llm = FakeLLMAdapter.timeouts_then(1)

    with pytest.raises(LLMTimeoutError):
        run_pipeline(
            _text("maria_ortiz.txt"),
            case_id="maria_ortiz",
            llm=llm,
            trace_dir=tmp_path,
        )
