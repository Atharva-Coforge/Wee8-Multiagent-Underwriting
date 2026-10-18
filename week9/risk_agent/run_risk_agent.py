"""Container entrypoint. Score ``EnrichedCase`` JSON files and print one JSON line each.

Runs only the risk scoring agent on Bedrock, with the week 8 retry on timeout.
Each result goes to stdout, which ECS sends to CloudWatch Logs. Nothing is
written to disk, so the container can run with a read-only root filesystem.

Exit codes: 0 every case scored, 1 a case escalated, 2 an input file is
missing or is not a valid ``EnrichedCase``. Any other error, such as
AccessDenied from Bedrock, is not caught: the traceback goes to the logs and
the task stops with a non-zero exit code.

Usage:
    python -m week9.risk_agent.run_risk_agent week9/fixtures/tyler_brandt.json
    python -m week9.risk_agent.run_risk_agent week9/fixtures/*.json
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from underwriting.agents._llm_json import AgentParseError
from underwriting.agents.result import AgentResult
from underwriting.agents.risk_scoring_agent import risk_scoring_agent
from underwriting.llm_adapter import LLMAdapter
from underwriting.models import EnrichedCase, RiskAssessment, TokenUsage
from underwriting.pipeline.retry import MAX_ATTEMPTS, retry_on_timeout
from week9.risk_agent.bedrock_adapter import BedrockAdapter

_TIMEOUT_REASON = (
    f"risk scoring timed out after {MAX_ATTEMPTS} attempts; sent to human review"
)


def main(
    argv: list[str] | None = None,
    *,
    llm: LLMAdapter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(
        description="Score EnrichedCase JSON files with the risk scoring agent."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="EnrichedCase JSON files")
    args = parser.parse_args(argv)

    cases: list[EnrichedCase] = []
    for path in args.inputs:
        try:
            cases.append(EnrichedCase.model_validate_json(path.read_text()))
        except (OSError, ValidationError) as exc:
            print(f"cannot read {path}: {exc}", file=sys.stderr)
            return 2

    adapter = llm if llm is not None else BedrockAdapter()
    exit_code = 0
    for enriched in cases:
        record = score_case(enriched, llm=adapter, sleep=sleep)
        print(json.dumps(record), flush=True)
        if record["status"] != "scored":
            exit_code = 1
    return exit_code


def score_case(
    enriched: EnrichedCase,
    *,
    llm: LLMAdapter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Score one case. Three timeouts or a parse failure escalate it to refer."""
    started = time.perf_counter()
    record: dict[str, Any] = {
        "case_id": enriched.intake.case_id,
        "model": getattr(llm, "model_id", None),
    }
    result: AgentResult[RiskAssessment] | None = None
    attempts = 0
    try:
        for attempt in retry_on_timeout(
            lambda _number: risk_scoring_agent(enriched, llm=llm),
            sleep=sleep,
        ):
            attempts = attempt.attempt
            if attempt.error is None:
                result = attempt.value
    except AgentParseError as exc:
        usage = exc.usage if exc.usage is not None else TokenUsage()
        record.update(_escalated(str(exc), attempts + 1, usage, exc.llm_calls, started))
        return record
    if result is None:
        record.update(_escalated(_TIMEOUT_REASON, attempts, TokenUsage(), 0, started))
        return record

    assessment = result.output
    record.update(
        {
            "status": "scored",
            "score": assessment.score,
            "tier": assessment.tier,
            "factors": [factor.model_dump() for factor in assessment.factors],
            "attempts": attempts,
            **result.usage.model_dump(),
            "llm_calls": result.llm_calls,
            "duration_ms": _elapsed_ms(started),
        }
    )
    return record


def _escalated(
    reason: str,
    attempts: int,
    usage: TokenUsage,
    llm_calls: int,
    started: float,
) -> dict[str, Any]:
    return {
        "status": "escalated",
        "decision": "refer",
        "reason": reason,
        "attempts": attempts,
        **usage.model_dump(),
        "llm_calls": llm_calls,
        "duration_ms": _elapsed_ms(started),
    }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    sys.exit(main())
