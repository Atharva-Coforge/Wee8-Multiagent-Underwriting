"""Freeze the risk scoring inputs in ``week9/fixtures/``.

Runs the week 8 intake and enrichment agents once on local Ollama for the five
cases that reach risk scoring, and saves each ``EnrichedCase`` as JSON. That is
the message enrichment would send to the cloud risk scoring agent.

Each fixture is checked against the enrichment span in ``traces/<case>.jsonl``
so it matches the input the on-prem risk score was produced from.

Usage:
    uv run python week9/build_fixtures.py
    OLLAMA_HOST=http://localhost:11434 uv run python week9/build_fixtures.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from underwriting.agents._llm_json import AgentParseError
from underwriting.agents.enrichment_agent import enrichment_agent
from underwriting.agents.errors import AgentEscalation
from underwriting.agents.intake_agent import intake_agent
from underwriting.llm_adapter import OllamaAdapter
from underwriting.models import EnrichedCase

_ROOT = Path(__file__).resolve().parents[1]
_APPLICATIONS = _ROOT / "data" / "applications"
_TRACES = _ROOT / "traces"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_OLLAMA_HOST = "http://host.docker.internal:11434"

# The cases whose week 8 run reached risk scoring.
CASES = (
    "maria_ortiz",
    "tyler_brandt",
    "clean_claim_mismatch",
    "messy_formats",
    "timeout_case",
)

# Enrichment span fields compared with the regenerated EnrichedCase.
_CHECKED = (
    "match_status",
    "person_id",
    "claims_band",
    "violation_severity",
    "vehicle_risk_band",
    "prior_insurance_band",
    "discrepancy_flags",
    "data_gaps",
)


def main() -> int:
    llm = OllamaAdapter(host=os.getenv("OLLAMA_HOST") or _OLLAMA_HOST)
    _FIXTURES.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for case_id in CASES:
        raw_text = (_APPLICATIONS / f"{case_id}.txt").read_text()
        try:
            intake = intake_agent(raw_text, case_id=case_id, llm=llm)
            enriched = enrichment_agent(intake.output, llm=llm).output
        except (AgentEscalation, AgentParseError) as exc:
            print(f"{case_id}: stopped before risk scoring: {exc}")
            failed.append(case_id)
            continue
        path = _FIXTURES / f"{case_id}.json"
        path.write_text(enriched.model_dump_json(indent=2) + "\n", encoding="utf-8")
        differences = _compare_with_trace(case_id, enriched)
        if differences:
            failed.append(case_id)
            print(f"{case_id}: wrote {path.name}, differs from trace")
            for line in differences:
                print(f"    {line}")
        else:
            print(f"{case_id}: wrote {path.name}, matches trace")
    if failed:
        print(f"Check these cases: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _compare_with_trace(case_id: str, enriched: EnrichedCase) -> list[str]:
    """Differences between the fixture and the traced enrichment output."""
    traced = _traced_enrichment(case_id)
    if traced is None:
        return ["no ok enrichment span in the trace"]
    current = _summary(enriched)
    return [
        f"{field}: trace {traced.get(field)!r}, fixture {current[field]!r}"
        for field in _CHECKED
        if traced.get(field) != current[field]
    ]


def _traced_enrichment(case_id: str) -> dict[str, Any] | None:
    path = _TRACES / f"{case_id}.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        span = json.loads(line)
        if span["agent"] == "enrichment" and span["status"] == "ok":
            return span["output"]
    return None


def _summary(enriched: EnrichedCase) -> dict[str, Any]:
    return {
        "match_status": enriched.lookup.match_status,
        "person_id": enriched.lookup.person_id,
        "claims_band": enriched.claims_band,
        "violation_severity": enriched.violation_severity,
        "vehicle_risk_band": enriched.vehicle_risk_band,
        "prior_insurance_band": enriched.prior_insurance_band,
        "discrepancy_flags": list(enriched.discrepancy_flags),
        "data_gaps": list(enriched.data_gaps),
    }


if __name__ == "__main__":
    sys.exit(main())
