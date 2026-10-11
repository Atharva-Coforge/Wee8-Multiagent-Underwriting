"""Intake's output is the object enrichment receives."""

import json
from pathlib import Path

from tests.fakes import FakeLLMAdapter

from underwriting.agents.enrichment_agent import enrichment_agent
from underwriting.models import EnrichedCase, IntakeRecord
from underwriting.pipeline.chain import run_pipeline

_APPLICATIONS = Path(__file__).resolve().parents[2] / "data" / "applications"


def test_enrichment_receives_the_intake_output(monkeypatch, tmp_path: Path):
    received: dict[str, IntakeRecord] = {}
    real = enrichment_agent

    def spy(intake: IntakeRecord, *, llm, database_dir=None):
        received["intake"] = intake
        return real(intake, llm=llm, database_dir=database_dir)

    monkeypatch.setattr("underwriting.pipeline.chain.enrichment_agent", spy)
    llm = FakeLLMAdapter(
        [
            {
                "full_name": "Maria Elena Ortiz",
                "date_of_birth": "1984-05-12",
                "vehicle_year": 2021,
                "vehicle_make": "Honda",
                "vehicle_model": "CR-V",
                "coverage": "100/300/100",
                "occupation": "registered nurse",
                "applicant_claims": "insured continuously; no accidents or tickets",
                "missing_fields": [],
                "decision": "proceed",
                "reason": "all required fields are present",
            },
            {
                "claims_band": "low",
                "violation_severity": "low",
                "vehicle_risk_band": "low",
                "prior_insurance_band": "low",
                "discrepancy_flags": [],
                "summary": "No at-fault accidents and no violations.",
            },
            {
                "score": 18,
                "factors": [
                    {"name": "claims_band low", "direction": "decrease", "weight": "low"},
                    {"name": "violation_severity low", "direction": "decrease", "weight": "low"},
                ],
            },
            {
                "decision": "approve",
                "rationale": "A score of 18 is a low tier.",
                "conditions": [],
            },
        ]
    )

    result = run_pipeline(
        (_APPLICATIONS / "maria_ortiz.txt").read_text(),
        case_id="maria_ortiz",
        llm=llm,
        trace_dir=tmp_path,
    )

    intake = result.agents[0].output
    enriched = result.agents[1].output
    assert isinstance(intake, IntakeRecord)
    assert isinstance(enriched, EnrichedCase)
    assert received["intake"] is intake
    assert enriched.intake is intake
    assert intake.full_name == "Maria Elena Ortiz"
    assert intake.date_of_birth == "1984-05-12"


def test_maria_trace_records_intake_then_enrichment(tmp_path: Path):
    llm = FakeLLMAdapter(
        [
            {
                "full_name": "Maria Elena Ortiz",
                "date_of_birth": "1984-05-12",
                "vehicle_year": 2021,
                "vehicle_make": "Honda",
                "vehicle_model": "CR-V",
                "coverage": "100/300/100",
                "occupation": "registered nurse",
                "applicant_claims": "insured continuously; no accidents or tickets",
                "missing_fields": [],
                "decision": "proceed",
                "reason": "all required fields are present",
            },
            {
                "claims_band": "low",
                "violation_severity": "low",
                "vehicle_risk_band": "low",
                "prior_insurance_band": "low",
                "discrepancy_flags": [],
                "summary": "No at-fault accidents and no violations.",
            },
            {
                "score": 18,
                "factors": [
                    {"name": "claims_band low", "direction": "decrease", "weight": "low"},
                    {"name": "violation_severity low", "direction": "decrease", "weight": "low"},
                ],
            },
            {
                "decision": "approve",
                "rationale": "A score of 18 is a low tier.",
                "conditions": [],
            },
        ]
    )

    run_pipeline(
        (_APPLICATIONS / "maria_ortiz.txt").read_text(),
        case_id="maria_ortiz",
        llm=llm,
        trace_dir=tmp_path,
    )

    path = tmp_path / "maria_ortiz.jsonl"
    assert path.is_file()
    assert list(tmp_path.glob("*.jsonl")) == [path]
    spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    intake_span, enrichment_span = spans[0], spans[1]
    assert intake_span["agent"] == "intake"
    assert enrichment_span["agent"] == "enrichment"
    assert enrichment_span["input"]["full_name"] == intake_span["output"]["full_name"]
    assert intake_span["output"]["full_name"] == "Maria Elena Ortiz"
    for span in (intake_span, enrichment_span):
        assert span["status"] == "ok"
        assert span["prompt_tokens"] > 0
        assert span["completion_tokens"] > 0
        assert span["total_tokens"] > 0
        assert span["duration_ms"] >= 0


def test_kevin_code_floor_override_is_one_rejected_intake_span(tmp_path: Path):
    llm = FakeLLMAdapter(
        [
            {
                "full_name": "Kevin Osei",
                "date_of_birth": None,
                "vehicle_year": 2019,
                "vehicle_make": "Ford",
                "vehicle_model": "F-150",
                "coverage": "100/300/100",
                "occupation": "electrician",
                "applicant_claims": "",
                "missing_fields": [],
                "decision": "proceed",
                "reason": "the applicant will call the date of birth in later",
            }
        ]
    )

    result = run_pipeline(
        (_APPLICATIONS / "incomplete.txt").read_text(),
        case_id="incomplete",
        llm=llm,
        trace_dir=tmp_path,
    )

    assert result.status == "rejected"
    assert result.decision == "reject"
    spans = [
        json.loads(line)
        for line in (tmp_path / "incomplete.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [span["agent"] for span in spans] == ["intake"]
    span = spans[0]
    assert span["status"] == "rejected"
    assert span["override"] is True
    assert span["output"]["record"]["model_decision"] == "proceed"
    assert "date of birth" in span["output"]["reason"]
