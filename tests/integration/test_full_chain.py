"""Full chain against a scripted fake. No Ollama."""

import json
from pathlib import Path

from tests.fakes import FakeLLMAdapter
from underwriting.llm_adapter import LLMTimeoutError
from underwriting.models import (
    EnrichedCase,
    IntakeRecord,
    Recommendation,
    RiskAssessment,
    TokenUsage,
)
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


def _timeouts(n: int) -> list[LLMTimeoutError]:
    return [LLMTimeoutError("timed out") for _ in range(n)]


def _no_sleep(delays: list[float]):
    def sleep(seconds: float) -> None:
        delays.append(seconds)

    return sleep


def _spans(tmp_path: Path, case_id: str) -> list[dict]:
    path = tmp_path / f"{case_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _denise_script(timeouts: int, then: list[dict] | None = None) -> list:
    replies: list = [
        _intake(
            "Denise Carol Whitfield",
            "1977-08-03",
            vehicle_year=2018,
            vehicle_make="Toyota",
            vehicle_model="Camry",
            occupation="office manager",
        ),
        _enrichment(summary="One minor speeding violation and a not-at-fault hail claim."),
        *_timeouts(timeouts),
    ]
    if then:
        replies.extend(then)
    return replies


def test_risk_scoring_recovers_after_two_timeouts(tmp_path: Path):
    slept: list[float] = []
    llm = FakeLLMAdapter(
        _denise_script(
            2,
            [
                _risk(18),
                _recommendation(),
            ],
        )
    )

    result = run_pipeline(
        _text("timeout_case.txt"),
        case_id="timeout_case",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep(slept),
    )

    assert result.status == "completed"
    assert result.decision == "approve"
    assert result.rationale == "A score of 18 is a low tier."
    assert [step.agent for step in result.agents] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "risk_scoring",
        "risk_scoring",
        "recommendation",
    ]
    risk_steps = [step for step in result.agents if step.agent == "risk_scoring"]
    assert [step.attempt for step in risk_steps] == [1, 2, 3]
    assert risk_steps[0].error == "timed out"
    assert risk_steps[1].error == "timed out"
    assert risk_steps[0].output is None
    assert risk_steps[1].output is None
    assert risk_steps[0].usage.total_tokens == 0
    assert isinstance(risk_steps[2].output, RiskAssessment)
    assert risk_steps[2].output.score == 18
    assert risk_steps[2].error is None
    assert isinstance(result.agents[-1].output, Recommendation)
    assert llm.call_count == 6
    assert slept == [1, 2]

    spans = _spans(tmp_path, "timeout_case")
    assert [span["agent"] for span in spans] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "risk_scoring",
        "risk_scoring",
        "recommendation",
    ]
    risk_spans = [span for span in spans if span["agent"] == "risk_scoring"]
    assert [span["attempt"] for span in risk_spans] == [1, 2, 3]
    assert [span["status"] for span in risk_spans] == ["error", "error", "ok"]
    assert risk_spans[0]["output"] is None
    assert risk_spans[1]["output"] is None
    assert risk_spans[0]["error"] == "timed out"
    assert risk_spans[1]["error"] == "timed out"
    assert risk_spans[0]["prompt_tokens"] == 0
    assert risk_spans[0]["completion_tokens"] == 0
    assert risk_spans[0]["total_tokens"] == 0
    assert risk_spans[2]["error"] is None
    assert risk_spans[2]["output"]["score"] == 18
    assert spans[-1]["agent"] == "recommendation"
    assert spans[-1]["status"] == "ok"


def test_risk_scoring_escalates_after_three_timeouts(tmp_path: Path):
    slept: list[float] = []
    llm = FakeLLMAdapter(_denise_script(3))

    result = run_pipeline(
        _text("timeout_case.txt"),
        case_id="timeout_case",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep(slept),
    )

    assert result.status == "escalated"
    assert result.decision == "refer"
    assert result.rationale == (
        "risk scoring timed out after 3 attempts; sent to human review"
    )
    assert [step.agent for step in result.agents] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "risk_scoring",
        "risk_scoring",
    ]
    for step in result.agents:
        assert not isinstance(step.output, RiskAssessment)
        assert not isinstance(step.output, Recommendation)
        assert getattr(step.output, "score", None) is None
    risk_steps = [step for step in result.agents if step.agent == "risk_scoring"]
    assert [step.attempt for step in risk_steps] == [1, 2, 3]
    assert all(step.output is None for step in risk_steps)
    assert all(step.error == "timed out" for step in risk_steps)
    assert all(step.usage.total_tokens == 0 for step in risk_steps)
    assert llm.call_count == 5
    assert slept == [1, 2]

    spans = _spans(tmp_path, "timeout_case")
    assert [span["agent"] for span in spans] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "risk_scoring",
        "risk_scoring",
    ]
    assert [span["status"] for span in spans] == ["ok", "ok", "error", "error", "error"]
    assert [span["attempt"] for span in spans] == [1, 1, 1, 2, 3]
    assert "recommendation" not in [span["agent"] for span in spans]
    for span in spans[2:]:
        assert span["output"] is None
        assert span["error"] == "timed out"
        assert span["prompt_tokens"] == 0
        assert span["completion_tokens"] == 0
        assert span["total_tokens"] == 0
    for span in spans:
        output = span["output"]
        if isinstance(output, dict):
            assert "score" not in output


def test_timeout_span_keeps_usage_carried_on_the_exception(tmp_path: Path):
    carried = LLMTimeoutError("timed out")
    carried.usage = TokenUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5)
    carried.llm_calls = 1
    llm = FakeLLMAdapter(_denise_script(0) + [carried, *_timeouts(2)])

    result = run_pipeline(
        _text("timeout_case.txt"),
        case_id="timeout_case",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep([]),
    )

    assert result.status == "escalated"
    spans = [span for span in _spans(tmp_path, "timeout_case") if span["agent"] == "risk_scoring"]
    assert spans[0]["prompt_tokens"] == 4
    assert spans[0]["completion_tokens"] == 1
    assert spans[0]["total_tokens"] == 5
    assert spans[0]["llm_calls"] == 1
    assert spans[0]["output"] is None
    assert spans[1]["total_tokens"] == 0
    assert spans[2]["total_tokens"] == 0
    assert result.agents[2].usage.total_tokens == 5
    assert result.agents[3].usage.total_tokens == 0


def _maria_script(timeout_at: int) -> list:
    """Maria's four replies, with one timeout in place of the agent at ``timeout_at``."""
    replies: list = [
        _intake(
            "Maria Elena Ortiz",
            "1984-05-12",
            applicant_claims="insured continuously; no accidents or tickets",
        ),
        _enrichment(),
        _risk(18),
        _recommendation(),
    ]
    replies[timeout_at] = LLMTimeoutError("timed out")
    return replies[: timeout_at + 1]


def _assert_timeout_escalation(result, agent: str) -> None:
    assert result.status == "escalated"
    assert result.decision == "refer"
    assert result.rationale == f"{agent} timed out; sent to human review"
    step = result.agents[-1]
    assert step.agent == agent
    assert step.attempt == 1
    assert step.output is None
    assert step.error == "timed out"
    assert step.usage.total_tokens == 0


def _assert_error_span(span: dict, agent: str) -> None:
    assert span["agent"] == agent
    assert span["status"] == "error"
    assert span["output"] is None
    assert span["error"] == "timed out"
    assert span["attempt"] == 1
    assert span["prompt_tokens"] == 0
    assert span["completion_tokens"] == 0
    assert span["total_tokens"] == 0


def test_intake_timeout_escalates_without_retry(tmp_path: Path):
    slept: list[float] = []
    llm = FakeLLMAdapter(_maria_script(0))

    result = run_pipeline(
        _text("maria_ortiz.txt"),
        case_id="intake_timeout",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep(slept),
    )

    _assert_timeout_escalation(result, "intake")
    assert [step.agent for step in result.agents] == ["intake"]
    assert llm.call_count == 1
    assert slept == []
    spans = _spans(tmp_path, "intake_timeout")
    assert [span["agent"] for span in spans] == ["intake"]
    _assert_error_span(spans[-1], "intake")


def test_enrichment_timeout_escalates_without_retry(tmp_path: Path):
    slept: list[float] = []
    llm = FakeLLMAdapter(_maria_script(1))

    result = run_pipeline(
        _text("maria_ortiz.txt"),
        case_id="enrichment_timeout",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep(slept),
    )

    _assert_timeout_escalation(result, "enrichment")
    assert [step.agent for step in result.agents] == ["intake", "enrichment"]
    assert llm.call_count == 2
    assert slept == []
    spans = _spans(tmp_path, "enrichment_timeout")
    assert [span["agent"] for span in spans] == ["intake", "enrichment"]
    assert spans[0]["status"] == "ok"
    _assert_error_span(spans[-1], "enrichment")
    assert "risk_scoring" not in [span["agent"] for span in spans]
    assert "recommendation" not in [span["agent"] for span in spans]


def test_recommendation_timeout_escalates_without_retry(tmp_path: Path):
    slept: list[float] = []
    llm = FakeLLMAdapter(_maria_script(3))

    result = run_pipeline(
        _text("maria_ortiz.txt"),
        case_id="recommendation_timeout",
        llm=llm,
        trace_dir=tmp_path,
        sleep=_no_sleep(slept),
    )

    _assert_timeout_escalation(result, "recommendation")
    assert [step.agent for step in result.agents] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]
    assert result.agents[-1].output is None
    assert llm.call_count == 4
    assert slept == []
    spans = _spans(tmp_path, "recommendation_timeout")
    assert [span["agent"] for span in spans] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]
    assert [span["status"] for span in spans[:3]] == ["ok", "ok", "ok"]
    _assert_error_span(spans[-1], "recommendation")
