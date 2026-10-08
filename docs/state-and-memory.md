# State and memory

## Design decision: prompt chaining with message passing

The underwriting pipeline is a fixed sequence. Every application goes through the same four steps, in the same order:

1. Intake
2. Enrichment
3. Risk scoring
4. Recommendation (approve, deny, or refer to a human underwriter)

That fixed order is prompt chaining. Nothing in the pipeline chooses a different route, loops, or runs steps in parallel. Because the steps are already decided, the workflow does not need a graph framework or a hosted tracing product. LangGraph and LangSmith stay out of this project. The chain, the handoffs, and the traces are plain Python.

Data moves by message passing. Each agent is a function. The value it returns is the argument the next agent receives. Intake's output is enrichment's input, enrichment's output is risk scoring's input, and risk scoring's output is the recommendation agent's input. There is no shared case record that every agent reads and updates. Each function returns the prior message nested inside the new one.

Real calls use local Ollama, model `qwen3.5:9b`. Tests use a fake adapter and do not need Ollama.

## Contracts

- `intake_agent(raw_application, *, llm) -> IntakeRecord`. Reads the raw personal-auto application. Returns a case id, the raw application, normalized applicant, vehicle, and coverage, `missing_fields`, and intake notes.
- `enrichment_agent(intake, *, llm) -> EnrichedCase`. Reads an `IntakeRecord`. Returns that record plus inferred claims band, violation severity, vehicle risk band, territory factor, `data_gaps`, and notes. No dataset or data vendor was provided, so every supplemental fact is inferred and marked unverified. A missing history (`null`) is a data gap, not a clean record. An empty list means the applicant reported none.
- `risk_scoring_agent(enriched, *, llm) -> RiskAssessment`. Reads an `EnrichedCase`. Returns that case plus a score from 0 to 100, a tier (`low`, `moderate`, `high`, `severe`), and the factors behind the score. Large `data_gaps` push the score up and the tier toward refer, rather than a confident low score.
- `recommendation_agent(assessment, *, llm) -> Recommendation`. Reads a `RiskAssessment`. Returns that assessment plus `approve`, `deny`, or `refer`, a rationale, and any conditions.

## Required-field gate

Required fields are `applicant.full_name`, `applicant.date_of_birth`, `vehicle.year`, `vehicle.make`, `vehicle.model`, and `coverage_requested.liability_limits`. A field is missing if it is absent, `null`, or an empty string. If any are missing, the chain stops after intake and returns status `referred_incomplete`. It does not call the later agents.

## Timeout

Only risk scoring retries. A timeout is tried three times total. Each attempt is its own span. If all three time out, the result is status `escalated`, decision `refer`, and a rationale that names the timeout. The recommendation agent is not called. A timeout does not also run a JSON repair call.

## JSON repair

Invalid JSON is handled once per agent call: parse, and on failure send one repair prompt that includes the validation error. If the repair is still invalid, stop the chain with status `escalated`, decision `refer`, and a rationale that names the agent and the parse failure.
