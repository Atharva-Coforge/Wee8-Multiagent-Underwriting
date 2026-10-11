"""Four agents in order. Risk scoring retries on timeout, then escalates.

Intake, enrichment, and recommendation do not retry. One timeout there is
recorded and the case is referred.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from underwriting.agents._llm_json import AgentParseError
from underwriting.agents.enrichment_agent import enrichment_agent
from underwriting.agents.errors import AgentEscalation
from underwriting.agents.intake_agent import intake_agent
from underwriting.agents.recommendation_agent import recommendation_agent
from underwriting.agents.result import AgentResult
from underwriting.agents.risk_scoring_agent import risk_scoring_agent
from underwriting.llm_adapter import LLMAdapter, LLMTimeoutError
from underwriting.models import Decision, PipelineResult, PipelineStatus, TokenUsage
from underwriting.pipeline.retry import MAX_ATTEMPTS, Attempt, retry_on_timeout
from underwriting.tracing import (
    TraceStatus,
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

SummarizeOutput = Callable[[Any], dict[str, Any]]


_RISK_TIMEOUT_RATIONALE = (
    f"risk scoring timed out after {MAX_ATTEMPTS} attempts; sent to human review"
)


@dataclass
class AgentRun:
    """One agent call. ``output`` is its message, or the stop record.

    ``attempt`` is 1 except when risk scoring is tried again. ``error`` is the
    timeout message when that attempt failed.
    """

    agent: str
    usage: TokenUsage
    llm_calls: int
    duration_ms: float
    output: object | None = None
    stopped: bool = False
    attempt: int = 1
    error: str | None = None


def run_pipeline(
    raw_text: str,
    *,
    case_id: str,
    llm: LLMAdapter,
    trace_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    """Run intake, then enrichment, risk scoring, and recommendation.

    Each agent's output is the next agent's input. Spans are written once to
    ``<trace_dir>/<case_id>.jsonl``. ``case_id`` stays the given id, the
    application file stem, after a database match. The file is not renamed to
    ``person_id``.

    ``PipelineResult`` has no path field, so the path is not stored on the result.

    Risk scoring retries ``LLMTimeoutError`` up to three times. ``sleep`` waits
    between those attempts. If every attempt times out, the result is
    ``escalated`` / ``refer`` with no score, and recommendation is not called.

    A timeout from intake, enrichment, or recommendation is not retried. That
    agent records one error span and the result is ``escalated`` / ``refer``.
    """
    started = time.perf_counter()
    agents: list[AgentRun] = []
    tracer = Tracer(case_id) if trace_dir is None else Tracer(case_id, trace_dir)
    try:
        intake = _invoke_once(
            "intake",
            lambda: intake_agent(raw_text, case_id=case_id, llm=llm),
            agents,
            tracer,
            input_summary=summarize_intake_input(raw_text),
            summarize_output=summarize_intake_output,
        )
        enriched = _invoke_once(
            "enrichment",
            lambda: enrichment_agent(intake.output, llm=llm),
            agents,
            tracer,
            input_summary=summarize_enrichment_input(intake.output),
            summarize_output=summarize_enrichment_output,
        )
        risk_input = summarize_risk_input(enriched.output)

        def _risk_call(attempt: int) -> AgentResult:
            return _invoke(
                "risk_scoring",
                lambda: risk_scoring_agent(enriched.output, llm=llm),
                agents,
                tracer,
                input_summary=risk_input,
                summarize_output=summarize_risk_output,
                attempt=attempt,
            )

        assessment: AgentResult | None = None
        for attempt in retry_on_timeout(_risk_call, sleep=sleep):
            if attempt.error is not None:
                _record_timeout(
                    agents,
                    tracer,
                    agent="risk_scoring",
                    attempt=attempt,
                    input_summary=risk_input,
                )
                continue
            assessment = attempt.value
        if assessment is None:
            return _finish(
                status="escalated",
                decision="refer",
                rationale=_RISK_TIMEOUT_RATIONALE,
                case_id=case_id,
                agents=agents,
                started=started,
            )
        recommendation = _invoke_once(
            "recommendation",
            lambda: recommendation_agent(assessment.output, llm=llm),
            agents,
            tracer,
            input_summary=summarize_recommendation_input(assessment.output),
            summarize_output=summarize_recommendation_output,
        )
    except _TimeoutEscalation as exc:
        return _finish(
            status="escalated",
            decision="refer",
            rationale=f"{exc.agent} timed out; sent to human review",
            case_id=case_id,
            agents=agents,
            started=started,
        )
    except AgentEscalation as exc:
        return _finish(
            status=exc.status,
            decision=exc.decision,
            rationale=exc.reason,
            case_id=exc.case_id,
            agents=agents,
            started=started,
        )
    except AgentParseError as exc:
        return _finish(
            status="escalated",
            decision="refer",
            rationale=str(exc),
            case_id=case_id,
            agents=agents,
            started=started,
        )
    finally:
        tracer.write()

    decided = recommendation.output
    return _finish(
        status="completed",
        decision=decided.decision,
        rationale=decided.rationale,
        case_id=case_id,
        agents=agents,
        started=started,
    )


class _TimeoutEscalation(Exception):
    """Intake, enrichment, or recommendation timed out. Not used for risk scoring."""

    def __init__(self, agent: str) -> None:
        self.agent = agent
        super().__init__(f"{agent} timed out")


def _invoke_once(
    agent: str,
    call,
    agents: list[AgentRun],
    tracer: Tracer,
    *,
    input_summary: dict[str, Any],
    summarize_output: SummarizeOutput,
) -> AgentResult:
    """Call one agent a single time. A timeout is recorded and escalated.

    Risk scoring does not use this. Its ``LLMTimeoutError`` must reach
    ``retry_on_timeout`` uncaught.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        return _invoke(
            agent,
            call,
            agents,
            tracer,
            input_summary=input_summary,
            summarize_output=summarize_output,
        )
    except LLMTimeoutError as exc:
        _record_timeout(
            agents,
            tracer,
            agent=agent,
            attempt=Attempt(
                attempt=1,
                started_at=started_at,
                duration_ms=_elapsed_ms(started),
                error=exc,
            ),
            input_summary=input_summary,
        )
        raise _TimeoutEscalation(agent) from exc


def _invoke(
    agent: str,
    call,
    agents: list[AgentRun],
    tracer: Tracer,
    *,
    input_summary: dict[str, Any],
    summarize_output: SummarizeOutput,
    attempt: int = 1,
) -> AgentResult:
    """Time one agent and record its span. A stop is recorded, then re-raised.

    ``LLMTimeoutError`` is not recorded here. Risk scoring's retry records each
    timed-out attempt. ``_invoke_once`` records one timeout for the other agents.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        result = call()
    except AgentEscalation as exc:
        usage = exc.usage if exc.usage is not None else TokenUsage()
        duration_ms = _elapsed_ms(started)
        agents.append(
            AgentRun(
                agent=agent,
                usage=usage,
                llm_calls=exc.llm_calls,
                duration_ms=duration_ms,
                output=exc.record,
                stopped=True,
                attempt=attempt,
            )
        )
        _record(
            tracer,
            agent=agent,
            started_at=started_at,
            duration_ms=duration_ms,
            usage=usage,
            llm_calls=exc.llm_calls,
            status=exc.status,
            input_summary=input_summary,
            output_summary=summarize_stop(exc),
            override=_override(exc.record),
            error=None,
            attempt=attempt,
        )
        raise
    except AgentParseError as exc:
        usage = exc.usage if exc.usage is not None else TokenUsage()
        duration_ms = _elapsed_ms(started)
        agents.append(
            AgentRun(
                agent=agent,
                usage=usage,
                llm_calls=exc.llm_calls,
                duration_ms=duration_ms,
                output=None,
                stopped=True,
                attempt=attempt,
            )
        )
        _record(
            tracer,
            agent=agent,
            started_at=started_at,
            duration_ms=duration_ms,
            usage=usage,
            llm_calls=exc.llm_calls,
            status="error",
            input_summary=input_summary,
            output_summary=None,
            override=False,
            error=str(exc),
            attempt=attempt,
        )
        raise
    usage = result.usage
    duration_ms = _elapsed_ms(started)
    agents.append(
        AgentRun(
            agent=agent,
            usage=usage,
            llm_calls=result.llm_calls,
            duration_ms=duration_ms,
            output=result.output,
            attempt=attempt,
        )
    )
    _record(
        tracer,
        agent=agent,
        started_at=started_at,
        duration_ms=duration_ms,
        usage=usage,
        llm_calls=result.llm_calls,
        status="ok",
        input_summary=input_summary,
        output_summary=summarize_output(result.output),
        override=_override(result.output),
        error=None,
        attempt=attempt,
    )
    return result


def _record_timeout(
    agents: list[AgentRun],
    tracer: Tracer,
    *,
    agent: str,
    attempt: Attempt[AgentResult],
    input_summary: dict[str, Any],
) -> None:
    """Record one timed-out attempt. No output is stored."""
    error = attempt.error
    assert error is not None
    usage, llm_calls = _usage_from_timeout(error)
    agents.append(
        AgentRun(
            agent=agent,
            usage=usage,
            llm_calls=llm_calls,
            duration_ms=attempt.duration_ms,
            output=None,
            attempt=attempt.attempt,
            error=str(error),
        )
    )
    _record(
        tracer,
        agent=agent,
        started_at=attempt.started_at,
        duration_ms=attempt.duration_ms,
        usage=usage,
        llm_calls=llm_calls,
        status="error",
        input_summary=input_summary,
        output_summary=None,
        override=False,
        error=str(error),
        attempt=attempt.attempt,
    )


def _usage_from_timeout(exc: LLMTimeoutError) -> tuple[TokenUsage, int]:
    """Zero tokens, unless ``exc`` carries a ``TokenUsage``."""
    raw = getattr(exc, "usage", None)
    if isinstance(raw, TokenUsage):
        calls = getattr(exc, "llm_calls", 0)
        return raw, calls if isinstance(calls, int) else 0
    return TokenUsage(), 0


def _record(
    tracer: Tracer,
    *,
    agent: str,
    started_at: str,
    duration_ms: float,
    usage: TokenUsage,
    llm_calls: int,
    status: TraceStatus,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any] | None,
    override: bool,
    error: str | None,
    attempt: int = 1,
) -> None:
    tracer.record(
        agent=agent,
        started_at=started_at,
        duration_ms=duration_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        llm_calls=llm_calls,
        status=status,
        input=input_summary,
        output=output_summary,
        override=override,
        error=error,
        attempt=attempt,
    )


def _override(record: object | None) -> bool:
    return bool(getattr(record, "override", False))


def _finish(
    *,
    status: PipelineStatus,
    decision: Decision,
    rationale: str,
    case_id: str,
    agents: list[AgentRun],
    started: float,
) -> PipelineResult:
    return PipelineResult(
        status=status,
        decision=decision,
        rationale=rationale,
        case_id=case_id,
        agents=agents,
        total_duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
