"""Write one JSONL span per agent call.

``models.Span`` is not the on-disk record. It nests token counts under
``token_usage`` and has no ``llm_calls`` field. Each span here is a plain dict
with flat token counts, and ``input`` / ``output`` are short summaries.
"""

import json
from pathlib import Path
from typing import Any, Literal

from underwriting.agents.errors import AgentEscalation
from underwriting.models import (
    EnrichedCase,
    IntakeRecord,
    LookupResult,
    Recommendation,
    RiskAssessment,
)

TraceStatus = Literal["ok", "rejected", "escalated", "error"]

_DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"


class Tracer:
    """In-memory spans for one case, written to ``trace_dir/<case_id>.jsonl``."""

    def __init__(self, case_id: str, trace_dir: Path = _DEFAULT_TRACE_DIR) -> None:
        self.case_id = case_id
        self.trace_dir = trace_dir
        self.spans: list[dict[str, Any]] = []

    def record(
        self,
        *,
        agent: str,
        started_at: str,
        duration_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        llm_calls: int,
        status: TraceStatus,
        input: dict[str, Any],
        output: dict[str, Any] | None,
        attempt: int = 1,
        override: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Append one span. ``started_at`` is ISO 8601 UTC."""
        span = {
            "agent": agent,
            "attempt": attempt,
            "started_at": started_at,
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "status": status,
            "input": input,
            "output": output,
            "override": override,
            "error": error,
        }
        self.spans.append(span)
        return span

    def write(self) -> Path:
        """Overwrite ``trace_dir/<case_id>.jsonl`` with one JSON object per line."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.case_id}.jsonl"
        lines = [json.dumps(span) for span in self.spans]
        text = "\n".join(lines)
        if text:
            text += "\n"
        path.write_text(text, encoding="utf-8")
        return path


def summarize_intake_input(raw_text: str) -> dict[str, Any]:
    return {"raw_text_chars": len(raw_text)}


def summarize_intake_output(record: IntakeRecord) -> dict[str, Any]:
    return {
        "decision": record.decision,
        "missing_fields": list(record.missing_fields),
        "full_name": record.full_name,
        "vehicle": _vehicle(record.vehicle_year, record.vehicle_make, record.vehicle_model),
        "coverage": record.coverage,
        "override": record.override,
        "model_decision": record.model_decision,
    }


def summarize_enrichment_input(intake: IntakeRecord) -> dict[str, Any]:
    return {
        "full_name": intake.full_name,
        "date_of_birth": intake.date_of_birth,
        "applicant_claims": intake.applicant_claims,
    }


def summarize_enrichment_output(record: EnrichedCase | LookupResult) -> dict[str, Any]:
    """Bands and flags for a finished case.

    A lookup stop has no bands. ``dob_mismatch`` (and any other lookup stop)
    keeps only ``match_status`` and ``person_id``, never the database rows.
    """
    if isinstance(record, LookupResult):
        return {
            "match_status": record.match_status,
            "person_id": record.person_id,
        }
    lookup = record.lookup
    return {
        "match_status": lookup.match_status,
        "person_id": lookup.person_id,
        "claims_band": record.claims_band,
        "violation_severity": record.violation_severity,
        "vehicle_risk_band": record.vehicle_risk_band,
        "prior_insurance_band": record.prior_insurance_band,
        "discrepancy_flags": list(record.discrepancy_flags),
        "data_gaps": list(record.data_gaps),
    }


def summarize_risk_input(enriched: EnrichedCase) -> dict[str, Any]:
    return {
        "claims_band": enriched.claims_band,
        "violation_severity": enriched.violation_severity,
        "vehicle_risk_band": enriched.vehicle_risk_band,
        "prior_insurance_band": enriched.prior_insurance_band,
        "flag_count": len(enriched.discrepancy_flags),
        "gap_count": len(enriched.data_gaps),
    }


def summarize_risk_output(assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "score": assessment.score,
        "tier": assessment.tier,
        "factor_names": [factor.name for factor in assessment.factors],
    }


def summarize_recommendation_input(assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "score": assessment.score,
        "tier": assessment.tier,
        "flag_count": len(assessment.enriched.discrepancy_flags),
    }


def summarize_recommendation_output(recommendation: Recommendation) -> dict[str, Any]:
    return {
        "decision": recommendation.decision,
        "override": recommendation.override,
        "model_decision": recommendation.model_decision,
        "conditions": list(recommendation.conditions),
    }


def summarize_stop(exc: AgentEscalation) -> dict[str, Any]:
    """The stop's status, decision, and reason, plus a summary of ``exc.record``."""
    summary: dict[str, Any] = {
        "status": exc.status,
        "decision": exc.decision,
        "reason": exc.reason,
    }
    if exc.record is not None:
        summary["record"] = _summarize_record(exc.record)
    return summary


def _summarize_record(record: object) -> dict[str, Any]:
    if isinstance(record, IntakeRecord):
        return summarize_intake_output(record)
    if isinstance(record, (LookupResult, EnrichedCase)):
        return summarize_enrichment_output(record)
    if isinstance(record, RiskAssessment):
        return summarize_risk_output(record)
    if isinstance(record, Recommendation):
        return summarize_recommendation_output(record)
    return {}


def _vehicle(year: int | None, make: str | None, model: str | None) -> str:
    parts: list[str] = []
    if year is not None:
        parts.append(str(year))
    if make:
        parts.append(make)
    if model:
        parts.append(model)
    return " ".join(parts)
