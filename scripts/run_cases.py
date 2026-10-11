"""Run application files through the underwriting chain on local Ollama.

Usage:
    uv run python scripts/run_cases.py maria_ortiz
    uv run python scripts/run_cases.py maria_ortiz tyler_brandt
    uv run python scripts/run_cases.py --all
    uv run python scripts/run_cases.py --inject-timeout 3 timeout_case
"""

import argparse
import sys
from pathlib import Path

from underwriting.llm_adapter import LLMAdapter, LLMResponse, LLMTimeoutError, OllamaAdapter
from underwriting.models import (
    EnrichedCase,
    IntakeRecord,
    PipelineResult,
    Recommendation,
    RiskAssessment,
)
from underwriting.pipeline.chain import AgentRun, run_pipeline

_ROOT = Path(__file__).resolve().parents[1]
_APPLICATIONS = _ROOT / "data" / "applications"
_OLLAMA_HOST = "http://host.docker.internal:11434"
_RISK_SYSTEM_PREFIX = "Score 0 to 100"


class _RiskScoringTimeouts:
    """Raise ``LLMTimeoutError`` on the first N risk-scoring completions.

    Later risk-scoring calls, and every other agent, use the wrapped adapter.
    The risk-scoring prompt is the only one that starts with ``Score 0 to 100``.
    """

    def __init__(self, inner: OllamaAdapter, times: int) -> None:
        self._inner = inner
        self._remaining = times

    def complete(self, *, system: str, user: str) -> LLMResponse:
        if self._remaining > 0 and system.startswith(_RISK_SYSTEM_PREFIX):
            self._remaining -= 1
            raise LLMTimeoutError("risk scoring timed out")
        return self._inner.complete(system=system, user=user)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run underwriting cases with the local Ollama model."
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="File stems in data/applications/, for example maria_ortiz",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every .txt file in data/applications/",
    )
    parser.add_argument(
        "--inject-timeout",
        type=int,
        default=0,
        metavar="N",
        help="Raise LLMTimeoutError on the first N risk-scoring calls, then call Ollama",
    )
    args = parser.parse_args(argv)

    if args.inject_timeout < 0:
        print("--inject-timeout must be 0 or greater.", file=sys.stderr)
        return 2

    available = _available_names()
    if args.all and args.names:
        print("Pass case names or --all, not both.", file=sys.stderr)
        return 2
    if args.all:
        names = available
    else:
        names = list(args.names)
    if not names:
        parser.error("name a case file stem, or pass --all")

    missing = [name for name in names if name not in available]
    if missing:
        listed = ", ".join(available)
        print(
            f"No application named {', '.join(missing)}. Available: {listed}",
            file=sys.stderr,
        )
        return 2

    llm: LLMAdapter = OllamaAdapter(host=_OLLAMA_HOST)
    if args.inject_timeout:
        llm = _RiskScoringTimeouts(llm, args.inject_timeout)
    for name in names:
        raw_text = (_APPLICATIONS / f"{name}.txt").read_text()
        result = run_pipeline(raw_text, case_id=name, llm=llm)
        _print_result(result)
    return 0


def _available_names() -> list[str]:
    return sorted(path.stem for path in _APPLICATIONS.glob("*.txt"))


def _print_result(result: PipelineResult) -> None:
    print(f"case {result.case_id}")
    for step in result.agents:
        print(_format_step(step))
    stopped = next((step for step in result.agents if step.stopped), None)
    if stopped is not None:
        print(f"  stopped at {stopped.agent}: {result.rationale}")
    elif result.status == "escalated":
        print(f"  escalated: {result.rationale}")
    print(
        f"  status {result.status}  decision {result.decision}  "
        f"total {_format_duration(result.total_duration_ms)}"
    )
    print(f"trace: traces/{result.case_id}.jsonl")
    print()


def _format_step(step: AgentRun) -> str:
    tail = f"{step.usage.total_tokens} tokens  {_format_duration(step.duration_ms)}"
    if step.error is not None:
        return (
            f"  {step.agent}: attempt {step.attempt}  "
            f"error: {step.error}  |  {tail}"
        )
    output = step.output
    if isinstance(output, IntakeRecord):
        return f"  intake: decision {output.decision}  |  {tail}"
    if isinstance(output, EnrichedCase):
        flags = ", ".join(output.discrepancy_flags) or "none"
        gaps = ", ".join(output.data_gaps) or "none"
        return (
            "  enrichment: "
            f"claims {output.claims_band}, "
            f"violations {output.violation_severity}, "
            f"vehicle {output.vehicle_risk_band}, "
            f"prior {output.prior_insurance_band}  |  "
            f"flags {flags}  |  gaps {gaps}  |  {tail}"
        )
    if isinstance(output, RiskAssessment):
        return (
            f"  risk_scoring: attempt {step.attempt}  "
            f"score {output.score}  tier {output.tier}  |  {tail}"
        )
    if isinstance(output, Recommendation):
        conditions = ", ".join(output.conditions) or "none"
        return (
            f"  recommendation: decision {output.decision}  "
            f"override {output.override}  |  {tail}\n"
            f"    rationale: {output.rationale}\n"
            f"    conditions: {conditions}"
        )
    if step.stopped:
        return f"  {step.agent}: stopped  |  {tail}"
    return f"  {step.agent}  |  {tail}"


def _format_duration(duration_ms: float) -> str:
    return f"{duration_ms / 1000:.2f}s"


if __name__ == "__main__":
    sys.exit(main())
