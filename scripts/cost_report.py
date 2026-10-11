"""Print cost and latency from ``traces/<case_id>.jsonl``.

Usage:
    uv run python scripts/cost_report.py
    uv run python scripts/cost_report.py --trace-dir traces
"""

import argparse
import sys
from pathlib import Path

from underwriting.cost import (
    HostedRate,
    PLACEHOLDER_RATES,
    case_totals,
    hosted_cost,
    is_completed,
    load_spans,
    per_agent_averages,
    worst_case_latency_ms,
)
from underwriting.llm_adapter import DEFAULT_TIMEOUT_SECONDS
from underwriting.pipeline.retry import BACKOFF_SECONDS

_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost and latency from trace JSONL files.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=_ROOT / "traces",
        help="Directory of <case_id>.jsonl files",
    )
    args = parser.parse_args(argv)
    print(render_report(args.trace_dir), end="")
    return 0


def render_report(trace_dir: Path) -> str:
    cases = load_spans(trace_dir)
    completed = {case_id: spans for case_id, spans in cases.items() if is_completed(spans)}
    incomplete = [case_id for case_id in cases if case_id not in completed]
    totals = {case_id: case_totals(spans) for case_id, spans in completed.items()}
    averages = per_agent_averages(cases)

    lines: list[str] = []
    lines.append(f"Completed cases: {len(completed)}")
    lines.append("")
    lines.append("Per-agent averages (completed cases only)")
    if averages:
        headers = ["agent", "prompt", "completion", "total", "ms"]
        rows = [
            [
                agent,
                f"{stats['prompt_tokens']:.1f}",
                f"{stats['completion_tokens']:.1f}",
                f"{stats['total_tokens']:.1f}",
                f"{stats['duration_ms']:.1f}",
            ]
            for agent, stats in averages.items()
        ]
        lines.append(_table(headers, rows))
    else:
        lines.append("  no completed trace")
    lines.append("")

    lines.append("Per case (completed traces only)")
    if totals:
        token_values = [item["total_tokens"] for item in totals.values()]
        second_values = [item["duration_ms"] / 1000 for item in totals.values()]
        prompt_avg = _mean([item["prompt_tokens"] for item in totals.values()])
        completion_avg = _mean([item["completion_tokens"] for item in totals.values()])
        normal_ms = _mean([item["duration_ms"] for item in totals.values()])
        lines.append(f"  average tokens: {_mean(token_values):.1f}")
        lines.append(f"  maximum tokens: {max(token_values)}")
        lines.append(f"  average seconds: {_mean(second_values):.2f}")
        lines.append(f"  maximum seconds: {max(second_values):.2f}")
    else:
        prompt_avg = 0.0
        completion_avg = 0.0
        normal_ms = 0.0
        lines.append("  no completed trace")
    lines.append("")

    lines.append("Local cost: $0")
    lines.append("")
    lines.append("Hosted cost")
    for name, rate in PLACEHOLDER_RATES.items():
        lines.append(f"  {name}: {_format_hosted(prompt_avg, completion_avg, rate, has_cases=bool(totals))}")
    lines.append("")

    backoff_s = float(sum(BACKOFF_SECONDS))
    worst_ms = worst_case_latency_ms(
        normal_ms,
        DEFAULT_TIMEOUT_SECONDS,
        backoff_s=backoff_s,
    )
    lines.append("Worst-case latency")
    lines.append(f"  timeout: {DEFAULT_TIMEOUT_SECONDS:g}s (DEFAULT_TIMEOUT_SECONDS)")
    lines.append(f"  backoff: {backoff_s:g}s")
    if totals:
        lines.append(f"  normal case: {normal_ms / 1000:.2f}s (average completed case)")
    else:
        lines.append("  normal case: no completed trace")
    lines.append(
        f"  worst case: {worst_ms / 1000:.2f}s "
        "(normal case + 3 timeouts + backoff)"
    )
    lines.append("")

    lines.append("Cases with no completed trace (not averaged)")
    if incomplete:
        for case_id in incomplete:
            lines.append(f"  {case_id}")
    else:
        lines.append("  none")
    lines.append("")
    return "\n".join(lines)


def _format_hosted(
    prompt_tokens: float,
    completion_tokens: float,
    rate: HostedRate | None,
    *,
    has_cases: bool,
) -> str:
    if rate is None:
        return "rate not set"
    if not has_cases:
        return "no completed trace"
    per_case = hosted_cost(prompt_tokens, completion_tokens, rate)
    return f"${per_case:.6f} per case, ${per_case * 1000:.4f} per 1,000 cases"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [format_row(headers), format_row(["-" * width for width in widths])]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
