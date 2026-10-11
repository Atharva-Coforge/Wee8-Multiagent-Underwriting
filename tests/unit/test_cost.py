"""Cost and latency math from synthetic JSONL traces. No network and no Ollama."""

import json
from pathlib import Path

from underwriting.cost import (
    PLACEHOLDER_RATES,
    case_totals,
    hosted_cost,
    load_spans,
    per_agent_averages,
    worst_case_latency_ms,
)

_AGENTS = ("intake", "enrichment", "risk_scoring", "recommendation")


def _span(
    agent: str,
    *,
    prompt: int,
    completion: int,
    duration_ms: float,
    status: str = "ok",
    llm_calls: int = 1,
) -> dict:
    return {
        "agent": agent,
        "duration_ms": duration_ms,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "llm_calls": llm_calls,
        "status": status,
    }


def _write_trace(trace_dir: Path, case_id: str, spans: list[dict]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(span) for span in spans)
    (trace_dir / f"{case_id}.jsonl").write_text(body + "\n\n", encoding="utf-8")


def _completed(prompts: tuple[int, int, int, int], *, duration_ms: float) -> list[dict]:
    return [
        _span(agent, prompt=prompt, completion=prompt // 10, duration_ms=duration_ms)
        for agent, prompt in zip(_AGENTS, prompts, strict=True)
    ]


def test_totals_add_up(tmp_path: Path):
    spans = [
        _span("intake", prompt=10, completion=5, duration_ms=100.5, llm_calls=1),
        _span("enrichment", prompt=4, completion=6, duration_ms=50, llm_calls=2),
    ]
    _write_trace(tmp_path, "maria_ortiz", spans)

    loaded = load_spans(tmp_path)

    assert list(loaded) == ["maria_ortiz"]
    assert len(loaded["maria_ortiz"]) == 2
    assert case_totals(loaded["maria_ortiz"]) == {
        "prompt_tokens": 14,
        "completion_tokens": 11,
        "total_tokens": 25,
        "llm_calls": 3,
        "duration_ms": 150.5,
    }


def test_incomplete_cases_are_excluded_from_averages(tmp_path: Path):
    _write_trace(tmp_path, "completed_a", _completed((100, 200, 300, 400), duration_ms=10))
    _write_trace(tmp_path, "completed_b", _completed((300, 400, 500, 600), duration_ms=30))
    _write_trace(
        tmp_path,
        "incomplete",
        [_span("intake", prompt=99_999, completion=99_999, duration_ms=99_999, status="rejected")],
    )
    stopped = _completed((1000, 1000, 1000, 1000), duration_ms=1000)
    stopped[-1]["status"] = "escalated"
    _write_trace(tmp_path, "four_but_stopped", stopped)
    _write_trace(
        tmp_path,
        "two_ok",
        [
            _span("intake", prompt=5000, completion=5000, duration_ms=5000),
            _span("enrichment", prompt=5000, completion=5000, duration_ms=5000),
        ],
    )

    averages = per_agent_averages(load_spans(tmp_path))

    assert set(averages) == set(_AGENTS)
    assert averages["intake"] == {
        "prompt_tokens": 200.0,
        "completion_tokens": 20.0,
        "total_tokens": 220.0,
        "duration_ms": 20.0,
    }
    assert averages["risk_scoring"]["prompt_tokens"] == 400.0
    assert averages["recommendation"]["duration_ms"] == 20.0


def test_hosted_cost_uses_usd_per_million_tokens():
    rate = {"input": 3.0, "output": 15.0}

    assert hosted_cost(2_000_000, 500_000, rate) == 13.5


def test_none_rate_reads_rate_not_set():
    assert hosted_cost(2_000_000, 500_000, None) == "rate not set"
    assert PLACEHOLDER_RATES["hosted_model_A"] is None
    assert PLACEHOLDER_RATES["hosted_model_B"] is None


def test_worst_case_is_normal_case_plus_timeouts_plus_backoff():
    assert worst_case_latency_ms(40_000, 120) == 40_000 + 3 * 120_000
    assert worst_case_latency_ms(1_000, 30, timeouts=3, backoff_s=0.5) == 1_000 + 90_000 + 500
