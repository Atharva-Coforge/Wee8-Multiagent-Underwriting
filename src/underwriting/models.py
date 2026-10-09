"""Messages passed from one agent to the next."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Band = Literal["low", "moderate", "high", "severe"]
Decision = Literal["approve", "deny", "refer", "reject"]
RecommendationDecision = Literal["approve", "deny", "refer"]
PipelineStatus = Literal["completed", "rejected", "escalated"]
IntakeDecision = Literal["proceed", "reject", "escalate"]
MatchStatus = Literal["full_match", "dob_mismatch", "not_found"]
SpanStatus = Literal["ok", "error", "timeout", "escalated", "rejected"]

_DATE = r"^\d{4}-\d{2}-\d{2}$"


class IntakeRecord(BaseModel):
    """Normalized application. Required fields may be None when the text omitted them."""

    case_id: str
    raw_text: str
    full_name: str | None = None
    date_of_birth: str | None = Field(default=None, pattern=_DATE)
    vehicle_year: int | None = None
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    coverage: str | None = None
    occupation: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    decision: IntakeDecision
    reason: str
    model_decision: IntakeDecision | None = None
    override: bool = False
    applicant_claims: str = ""


class Accident(BaseModel):
    date: str = Field(pattern=_DATE)
    at_fault: bool
    description: str
    bodily_injury: bool
    amount_paid_usd: int


class Violation(BaseModel):
    date: str = Field(pattern=_DATE)
    type: str
    description: str


class Claim(BaseModel):
    date: str = Field(pattern=_DATE)
    type: str
    at_fault: bool
    description: str
    bodily_injury: bool
    amount_paid_usd: int


class DrivingHistoryRecord(BaseModel):
    person_id: str
    full_name: str
    date_of_birth: str = Field(pattern=_DATE)
    accidents: list[Accident]
    violations: list[Violation]
    claims: list[Claim]


class OwnedVehicle(BaseModel):
    year: int
    make: str
    model: str
    currently_insured: bool


class VehicleRecord(BaseModel):
    person_id: str
    full_name: str
    date_of_birth: str = Field(pattern=_DATE)
    vehicles: list[OwnedVehicle]


class Policy(BaseModel):
    carrier: str
    currently_insured: bool
    years_continuous: int
    lapse_days: int
    cancelled: bool
    non_renewed: bool


class PriorInsuranceRecord(BaseModel):
    person_id: str
    full_name: str
    date_of_birth: str = Field(pattern=_DATE)
    policies: list[Policy]


class LookupResult(BaseModel):
    """Rows found for one normalized name and date of birth."""

    match_status: MatchStatus
    person_id: str | None = None
    driving_history: DrivingHistoryRecord | None = None
    vehicles: VehicleRecord | None = None
    prior_insurance: PriorInsuranceRecord | None = None


class EnrichedCase(BaseModel):
    """Intake plus lookup. Database rows are verified."""

    intake: IntakeRecord
    lookup: LookupResult
    claims_band: Band
    violation_severity: Band
    vehicle_risk_band: Band
    discrepancy_flags: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    summary: str
    verified: Literal[True] = True


class RiskFactor(BaseModel):
    name: str
    direction: Literal["increase", "decrease"]
    weight: Literal["low", "moderate", "high"]


class RiskAssessment(BaseModel):
    enriched: EnrichedCase
    score: int = Field(ge=0, le=100)
    tier: Band
    factors: list[RiskFactor]


class Recommendation(BaseModel):
    """approve, deny, or refer. reject is an intake outcome, not a recommendation."""

    assessment: RiskAssessment
    decision: RecommendationDecision
    rationale: str
    conditions: list[str] = Field(default_factory=list)
    override: bool = False
    model_decision: RecommendationDecision | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Span(BaseModel):
    """One agent call. input and output are short summaries, not the nested message."""

    agent: str
    started_at: str
    duration_ms: float
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    token_usage: TokenUsage
    status: SpanStatus
    attempt: int = 1
    error: str | None = None
    override: bool = False


class Trace(BaseModel):
    trace_id: str
    case_id: str
    spans: list[Span] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Final outcome. Escalated and rejected cases have no recommendation."""

    status: PipelineStatus
    decision: Decision
    rationale: str
    case_id: str
    trace: Trace | None = None
