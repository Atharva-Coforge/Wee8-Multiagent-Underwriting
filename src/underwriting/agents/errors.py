"""Stop the chain when an agent rejects or escalates a case."""

from typing import Literal

from pydantic import BaseModel

from underwriting.models import TokenUsage

PipelineStopStatus = Literal["rejected", "escalated"]
PipelineStopDecision = Literal["reject", "refer"]


class AgentEscalation(Exception):
    """An agent ended the chain. The pipeline catches this.

    Timeouts and JSON parse failures use their own exceptions.
    """

    def __init__(
        self,
        *,
        status: PipelineStopStatus,
        decision: PipelineStopDecision,
        reason: str,
        case_id: str,
        record: BaseModel | None = None,
        usage: TokenUsage | None = None,
        llm_calls: int = 0,
    ) -> None:
        self.status = status
        self.decision = decision
        self.reason = reason
        self.case_id = case_id
        self.record = record
        self.usage = usage
        self.llm_calls = llm_calls
        super().__init__(reason)
