"""Tracer JSONL and the short span summaries."""

import json
from pathlib import Path

from underwriting.agents.errors import AgentEscalation
from underwriting.models import (
    Accident,
    DrivingHistoryRecord,
    EnrichedCase,
    IntakeRecord,
    LookupResult,
    Recommendation,
    RiskAssessment,
    RiskFactor,
)
from underwriting.tracing import (
    Tracer,
    summarize_enrichment_input,
    summarize_enrichment_output,
    summarize_intake_input,
    summarize_intake_output,
    summarize_recommendation_input,
    summarize_recommendation_output,
    summarize_risk_input,
    summarize_risk_output,
    summarize_stop,
)

_SPAN_KEYS = {
    "agent",
    "attempt",
    "started_at",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "llm_calls",
    "status",
    "input",
    "output",
    "override",
    "error",
}


def test_default_trace_dir_is_the_repo_traces_folder():
    tracer = Tracer("not-written")

    assert tracer.trace_dir == Path(__file__).resolve().parents[2] / "traces"
    assert tracer.spans == []


def test_record_and_write_jsonl_in_order(tmp_path: Path):
    trace_dir = tmp_path / "nested" / "traces"
    tracer = Tracer("maria_ortiz", trace_dir=trace_dir)
    tracer.record(
        agent="intake",
        started_at="2026-10-10T12:00:00+00:00",
        duration_ms=12.5,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        llm_calls=1,
        status="ok",
        input={"raw_text_chars": 40},
        output={"decision": "proceed", "full_name": "Maria Elena Ortiz"},
    )
    tracer.record(
        agent="enrichment",
        attempt=2,
        started_at="2026-10-10T12:00:01+00:00",
        duration_ms=3.0,
        prompt_tokens=4,
        completion_tokens=6,
        total_tokens=10,
        llm_calls=0,
        status="escalated",
        input={"full_name": "Maria Elena Ortiz"},
        output=None,
        error="possible identity mismatch",
    )

    path = tracer.write()

    assert path == trace_dir / "maria_ortiz.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert set(first) == _SPAN_KEYS
    assert set(second) == _SPAN_KEYS
    assert [first["agent"], second["agent"]] == ["intake", "enrichment"]
    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert first["output"]["full_name"] == "Maria Elena Ortiz"
    assert second["output"] is None
    assert second["error"] == "possible identity mismatch"
    assert first["prompt_tokens"] + first["completion_tokens"] == first["total_tokens"]
    assert second["prompt_tokens"] + second["completion_tokens"] == second["total_tokens"]
    assert first["total_tokens"] + second["total_tokens"] == 25
    assert (
        first["prompt_tokens"] + second["prompt_tokens"]
        + first["completion_tokens"] + second["completion_tokens"]
        == first["total_tokens"] + second["total_tokens"]
    )

    replacement = Tracer("maria_ortiz", trace_dir=trace_dir)
    replacement.record(
        agent="intake",
        started_at="2026-10-10T13:00:00+00:00",
        duration_ms=1.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        llm_calls=1,
        status="ok",
        input={"raw_text_chars": 1},
        output={"decision": "proceed"},
    )
    replacement.write()
    rewritten = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [span["agent"] for span in rewritten] == ["intake"]
    assert rewritten[0]["total_tokens"] == 2


def test_summary_helpers_return_expected_keys():
    intake = _intake_record()
    assert summarize_intake_input("hello") == {"raw_text_chars": 5}

    intake_output = summarize_intake_output(intake)
    assert set(intake_output) == {
        "decision",
        "missing_fields",
        "full_name",
        "vehicle",
        "coverage",
        "override",
        "model_decision",
    }
    assert intake_output["vehicle"] == "2021 Honda CR-V"
    assert intake_output["full_name"] == "Maria Elena Ortiz"

    enrichment_input = summarize_enrichment_input(intake)
    assert set(enrichment_input) == {"full_name", "date_of_birth", "applicant_claims"}
    assert enrichment_input["applicant_claims"] == "no accidents"

    case = _enriched(intake)
    enrichment_output = summarize_enrichment_output(case)
    assert set(enrichment_output) == {
        "match_status",
        "person_id",
        "claims_band",
        "violation_severity",
        "vehicle_risk_band",
        "prior_insurance_band",
        "discrepancy_flags",
        "data_gaps",
    }
    assert "driving_history" not in enrichment_output
    assert "intake" not in enrichment_output

    risk_input = summarize_risk_input(case)
    assert set(risk_input) == {
        "claims_band",
        "violation_severity",
        "vehicle_risk_band",
        "prior_insurance_band",
        "flag_count",
        "gap_count",
    }
    assert risk_input["flag_count"] == 1
    assert risk_input["gap_count"] == 1

    assessment = RiskAssessment(
        enriched=case,
        score=42,
        tier="moderate",
        factors=[RiskFactor(name="claims_band low", direction="decrease", weight="low")],
    )
    risk_output = summarize_risk_output(assessment)
    assert set(risk_output) == {"score", "tier", "factor_names"}
    assert risk_output["factor_names"] == ["claims_band low"]

    recommendation_input = summarize_recommendation_input(assessment)
    assert set(recommendation_input) == {"score", "tier", "flag_count"}
    assert recommendation_input["flag_count"] == 1

    recommendation = Recommendation(
        assessment=assessment,
        decision="refer",
        rationale="A discrepancy flag blocks approval.",
        conditions=[],
        override=True,
        model_decision="approve",
    )
    recommendation_output = summarize_recommendation_output(recommendation)
    assert set(recommendation_output) == {
        "decision",
        "override",
        "model_decision",
        "conditions",
    }
    assert recommendation_output["model_decision"] == "approve"

    stopped = AgentEscalation(
        status="rejected",
        decision="reject",
        reason="required information date of birth is missing",
        case_id="incomplete",
        record=intake,
    )
    stop = summarize_stop(stopped)
    assert set(stop) == {"status", "decision", "reason", "record"}
    assert set(stop["record"]) == set(intake_output)
    assert stop["reason"] == "required information date of birth is missing"

    without_record = AgentEscalation(
        status="escalated",
        decision="refer",
        reason="insufficient information found",
        case_id="unknown_person",
    )
    assert set(summarize_stop(without_record)) == {"status", "decision", "reason"}


def test_dob_mismatch_enrichment_summary_omits_database_rows():
    other_person = DrivingHistoryRecord(
        person_id="P-1004",
        full_name="Jordan A. Washington",
        date_of_birth="1988-02-02",
        accidents=[
            Accident(
                date="2024-01-08",
                at_fault=True,
                description="SECRET-OTHER-PERSON-ACCIDENT",
                bodily_injury=True,
                amount_paid_usd=42000,
            )
        ],
        violations=[],
        claims=[],
    )
    lookup = LookupResult(
        match_status="dob_mismatch",
        person_id="P-1004",
        driving_history=other_person,
    )

    summary = summarize_enrichment_output(lookup)

    assert summary == {"match_status": "dob_mismatch", "person_id": "P-1004"}
    dumped = json.dumps(summary)
    assert "SECRET-OTHER-PERSON-ACCIDENT" not in dumped
    assert "accidents" not in summary
    assert "driving_history" not in summary
    assert "Jordan" not in dumped


def _intake_record() -> IntakeRecord:
    return IntakeRecord(
        case_id="maria_ortiz",
        raw_text="Maria Elena Ortiz applies for coverage.",
        full_name="Maria Elena Ortiz",
        date_of_birth="1984-05-12",
        vehicle_year=2021,
        vehicle_make="Honda",
        vehicle_model="CR-V",
        coverage="100/300/100",
        occupation="registered nurse",
        missing_fields=[],
        decision="proceed",
        reason="all required fields are present",
        applicant_claims="no accidents",
    )


def _enriched(intake: IntakeRecord) -> EnrichedCase:
    return EnrichedCase(
        intake=intake,
        lookup=LookupResult(match_status="full_match", person_id="P-1001"),
        claims_band="low",
        violation_severity="low",
        vehicle_risk_band="low",
        prior_insurance_band="low",
        discrepancy_flags=["applicant claims no accidents; history shows one"],
        data_gaps=["no prior insurance record found"],
        summary="Clean bands with one discrepancy flag.",
    )
