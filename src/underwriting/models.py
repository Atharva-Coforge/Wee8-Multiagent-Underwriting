"""Messages passed from one agent to the next."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Band = Literal["low", "moderate", "high", "severe"]
Decision = Literal["approve", "deny", "refer"]
PipelineStatus = Literal["completed", "referred_incomplete", "escalated"]
Loose = str | int | float | bool | None


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class RawAddress(_Loose):
    city: str | None = None
    state: str | None = None
    zip: str | None = None


class RawLicense(_Loose):
    status: str | None = None
    first_licensed_date: str | None = None


class RawApplicant(_Loose):
    full_name: str | None = None
    date_of_birth: str | None = None
    address: RawAddress | None = None
    license: RawLicense | None = None


class RawVehicle(_Loose):
    year: Loose = None
    make: str | None = None
    model: str | None = None
    primary_use: str | None = None
    annual_mileage: Loose = None
    garaging_zip: str | None = None


class RawCoverage(_Loose):
    liability_limits: str | None = None


class RawPriorInsurance(_Loose):
    currently_insured: Loose = None
    years_continuous: Loose = None
    lapse_days: Loose = None


class RawAccident(_Loose):
    date: str | None = None
    at_fault: Loose = None
    description: str | None = None
    bodily_injury: Loose = None
    amount_paid_usd: Loose = None


class RawViolation(_Loose):
    date: str | None = None
    type: str | None = None
    description: str | None = None


class RawClaim(_Loose):
    date: str | None = None
    type: str | None = None
    at_fault: Loose = None
    amount_paid_usd: Loose = None


class RawDrivingHistory(_Loose):
    accidents: list[RawAccident] | None = None
    violations: list[RawViolation] | None = None
    other_claims: list[RawClaim] | None = None


class RawApplication(_Loose):
    application_id: str | None = None
    applicant: RawApplicant | None = None
    vehicle: RawVehicle | None = None
    coverage_requested: RawCoverage | None = None
    prior_insurance: RawPriorInsurance | None = None
    driving_history: RawDrivingHistory | None = None
    applicant_notes: str | None = None


class Address(BaseModel):
    city: str | None = None
    state: str | None = None
    zip: str | None = None


class Applicant(BaseModel):
    full_name: str | None = None
    date_of_birth: str | None = None
    address: Address | None = None
    license_status: str | None = None
    first_licensed_date: str | None = None


class Vehicle(BaseModel):
    year: int | None = None
    make: str | None = None
    model: str | None = None
    primary_use: str | None = None
    annual_mileage: int | None = None
    garaging_zip: str | None = None


class Coverage(BaseModel):
    liability_limits: str | None = None


class PriorInsurance(BaseModel):
    currently_insured: bool | None = None
    years_continuous: int | None = None
    lapse_days: int | None = None


class Accident(BaseModel):
    date: str
    at_fault: bool
    description: str
    bodily_injury: bool
    amount_paid_usd: int


class Violation(BaseModel):
    date: str
    type: str
    description: str


class OtherClaim(BaseModel):
    date: str
    type: str
    at_fault: bool
    amount_paid_usd: int


class DrivingHistory(BaseModel):
    accidents: list[Accident] | None = None
    violations: list[Violation] | None = None
    other_claims: list[OtherClaim] | None = None


class IntakeRecord(BaseModel):
    application_id: str
    raw_application: RawApplication
    applicant: Applicant
    vehicle: Vehicle
    coverage_requested: Coverage
    prior_insurance: PriorInsurance
    driving_history: DrivingHistory
    applicant_notes: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    intake_notes: str = ""


class EnrichedCase(BaseModel):
    intake: IntakeRecord
    claims_band: Band
    violation_severity: Band
    vehicle_risk_band: Band
    territory_factor: Band
    data_gaps: list[str] = Field(default_factory=list)
    enrichment_notes: str = ""
    unverified: Literal[True] = True


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
    assessment: RiskAssessment
    decision: Decision
    rationale: str
    conditions: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Span(BaseModel):
    agent: str
    started_at: str
    duration_ms: float
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    token_usage: TokenUsage
    status: Literal["ok", "error"]
    attempt: int = 1
    error: str | None = None


class Trace(BaseModel):
    trace_id: str
    case_id: str
    spans: list[Span] = Field(default_factory=list)


class PipelineResult(BaseModel):
    status: PipelineStatus
    decision: Decision
    rationale: str
    trace: Trace | None = None
