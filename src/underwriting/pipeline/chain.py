"""Four agents in order. Retries come later."""

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
from underwriting.llm_adapter import LLMAdapter
from underwriting.models import Decision, PipelineResult, PipelineStatus, TokenUsage
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


@dataclass
class AgentRun:
    """One agent the chain called. ``output`` is its message, or the stop record."""

    agent: str
    usage: TokenUsage
    llm_calls: int
    duration_ms: float
    output: object | None = None
    stopped: bool = False


def run_pipeline(
    raw_text: str,
    *,
    case_id: str,
    llm: LLMAdapter,
    trace_dir: Path | None = None,
) -> PipelineResult:
    """Run intake, then enrichment, risk scoring, and recommendation.

    Each agent's output is the next agent's input. Spans are written once to
    ``<trace_dir>/<case_id>.jsonl``. ``case_id`` stays the given id, the
    application file stem, after a database match. The file is not renamed to
    ``person_id``.

    ``PipelineResult`` has no path field, so the path is not stored on the result.

    ``LLMTimeoutError`` propagates.
    TODO: retry risk scoring on LLMTimeoutError, then escalate. Do not catch it here.
    """
    started = time.perf_counter()
    agents: list[AgentRun] = []
    tracer = Tracer(case_id) if trace_dir is None else Tracer(case_id, trace_dir)
    try:
        intake = _invoke(
            "intake",
            lambda: intake_agent(raw_text, case_id=case_id, llm=llm),
            agents,
            tracer,
            input_summary=summarize_intake_input(raw_text),
            summarize_output=summarize_intake_output,
        )
        enriched = _invoke(
            "enrichment",
            lambda: enrichment_agent(intake.output, llm=llm),
            agents,
            tracer,
            input_summary=summarize_enrichment_input(intake.output),
            summarize_output=summarize_enrichment_output,
        )
        # TODO: retry this call on LLMTimeoutError (three attempts, then escalate).
        assessment = _invoke(
            "risk_scoring",
            lambda: risk_scoring_agent(enriched.output, llm=llm),
            agents,
            tracer,
            input_summary=summarize_risk_input(enriched.output),
            summarize_output=summarize_risk_output,
        )
        recommendation = _invoke(
            "recommendation",
            lambda: recommendation_agent(assessment.output, llm=llm),
            agents,
            tracer,
            input_summary=summarize_recommendation_input(assessment.output),
            summarize_output=summarize_recommendation_output,
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


def _invoke(
    agent: str,
    call,
    agents: list[AgentRun],
    tracer: Tracer,
    *,
    input_summary: dict[str, Any],
    summarize_output: SummarizeOutput,
) -> AgentResult:
    """Time one agent and record its span. A stop is recorded, then re-raised.

    TODO: LLMTimeoutError is not caught. The retry step will handle it.
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
    )
    return result


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
