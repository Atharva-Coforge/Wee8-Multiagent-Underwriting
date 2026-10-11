"""Cost and latency from trace spans. No I/O except reading JSONL files."""

import json
from pathlib import Path
from typing import Any, TypedDict

Span = dict[str, Any]

_PIPELINE_AGENTS = ("intake", "enrichment", "risk_scoring", "recommendation")


class CaseTotals(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int
    duration_ms: float


class AgentAverage(TypedDict):
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    duration_ms: float


class HostedRate(TypedDict):
    """USD per 1M tokens. ``input`` prices prompt tokens, ``output`` prices completion tokens."""

    input: float
    output: float


# USD per 1M tokens, from https://www.anthropic.com/pricing, checked 2026-10-11.
HOSTED_RATES: dict[str, HostedRate | None] = {
    "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
}


def load_spans(trace_dir: Path) -> dict[str, list[Span]]:
    """Read ``trace_dir/<case_id>.jsonl`` into a span list per case id."""
    cases: dict[str, list[Span]] = {}
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        spans: list[Span] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            spans.append(json.loads(line))
        cases[path.stem] = spans
    return cases


def case_totals(spans: list[Span]) -> CaseTotals:
    """Sum token counts, LLM calls, and duration for one case."""
    return {
        "prompt_tokens": sum(int(span["prompt_tokens"]) for span in spans),
        "completion_tokens": sum(int(span["completion_tokens"]) for span in spans),
        "total_tokens": sum(int(span["total_tokens"]) for span in spans),
        "llm_calls": sum(int(span["llm_calls"]) for span in spans),
        "duration_ms": float(sum(span["duration_ms"] for span in spans)),
    }


def is_completed(spans: list[Span]) -> bool:
    """A completed case has four spans and every status is ``ok``."""
    return len(spans) == 4 and all(span["status"] == "ok" for span in spans)


def per_agent_averages(cases: dict[str, list[Span]]) -> dict[str, AgentAverage]:
    """Average tokens and duration per agent, completed cases only."""
    buckets: dict[str, list[Span]] = {}
    for spans in cases.values():
        if not is_completed(spans):
            continue
        for span in spans:
            buckets.setdefault(str(span["agent"]), []).append(span)

    averages: dict[str, AgentAverage] = {}
    for agent in sorted(buckets, key=_agent_sort_key):
        group = buckets[agent]
        count = len(group)
        averages[agent] = {
            "prompt_tokens": sum(span["prompt_tokens"] for span in group) / count,
            "completion_tokens": sum(span["completion_tokens"] for span in group) / count,
            "total_tokens": sum(span["total_tokens"] for span in group) / count,
            "duration_ms": sum(span["duration_ms"] for span in group) / count,
        }
    return averages


def hosted_cost(
    prompt_tokens: float,
    completion_tokens: float,
    rate: HostedRate | None,
) -> float | str:
    """USD for one case. ``rate`` is USD per 1M tokens, or ``None`` when unset."""
    if rate is None:
        return "rate not set"
    return (
        prompt_tokens / 1_000_000 * rate["input"]
        + completion_tokens / 1_000_000 * rate["output"]
    )


def worst_case_latency_ms(
    normal_case_ms: float,
    timeout_s: float,
    timeouts: int = 3,
    backoff_s: float = 0,
) -> float:
    """Normal case time plus ``timeouts`` client timeouts plus the total backoff.

    This is the path where risk scoring times out three times and then escalates.
    ``timeout_s`` and ``backoff_s`` are seconds. ``backoff_s`` is the total wait
    for that path, added once. The result is milliseconds.
    """
    return normal_case_ms + timeouts * timeout_s * 1000 + backoff_s * 1000


def _agent_sort_key(agent: str) -> tuple[int, str]:
    try:
        return (_PIPELINE_AGENTS.index(agent), agent)
    except ValueError:
        return (len(_PIPELINE_AGENTS), agent)
