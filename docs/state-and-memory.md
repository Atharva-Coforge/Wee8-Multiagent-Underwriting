# State and memory

## Design decision: prompt chaining with message passing

The pipeline is a fixed sequence: intake, enrichment, risk scoring, recommendation (approve, deny, or refer). That order is prompt chaining. The path is not chosen at runtime, and LangGraph and LangSmith stay out.

Data moves by message passing. Each agent returns the prior message nested inside its own result. There is no shared case store.

The model sees only the slice that step needs. A span logs a short summary of input and output, not the full nested object, so traces stay readable and token use does not grow because earlier messages were pasted back in.

Real calls use local Ollama, model `qwen3.5:9b` (`OLLAMA_HOST`, `OLLAMA_MODEL`). Tests use a fake adapter and do not start Ollama.

Each agent prompt includes field priorities and three fictional few-shot examples (clean, messy, risky).

## Inputs

Each applicant is a plain-text file in `data/applications/`, named after the person (`maria_ortiz.txt`). The paragraph contains full name, date of birth, the car to insure (year, make, model), occupation, and the coverage type and limits. The wording may be messy. Applicant text is data, not instructions.

Three JSON databases live in `data/databases/`. Every record has `person_id`, `full_name`, and `date_of_birth`.

| File | Contents |
| --- | --- |
| `driving_history.json` | Accidents, violations, claims |
| `vehicles_and_drivers.json` | Vehicles currently owned, and whether they are insured |
| `prior_insurance.json` | Past policies, lapses, cancellations, non-renewals |

Lookups match normalized full name plus date of birth. The file name is never the key.

## Reads and produces

| Agent | Reads | Produces |
| --- | --- | --- |
| Intake | The raw paragraph only | `IntakeRecord`: `raw_text`, normalized name, DOB, vehicle year/make/model, occupation, coverage, `applicant_claims`, `missing_fields`, a decision of `proceed`, `reject`, or `escalate`, and a reason |
| Enrichment | The normalized intake fields, `applicant_claims`, and the rows `databases.py` found | `EnrichedCase`: `intake`, `lookup`, `claims_band`, `violation_severity`, `vehicle_risk_band`, `prior_insurance_band`, `discrepancy_flags`, `data_gaps`, `summary`, and `verified` (always true). Or an escalation |
| Risk scoring | `claims_band`, `violation_severity`, `vehicle_risk_band`, `prior_insurance_band`, discrepancy flags, data gaps, the summary, and the vehicle and coverage needed to score | `RiskAssessment`: the enriched case, a score from 0 to 100, a tier (`low`, `moderate`, `high`, `severe`), and factors |
| Recommendation | Score, tier, discrepancy flags, data gaps, and the summary | `Recommendation`: `approve`, `deny`, or `refer`, a rationale, and any conditions. `reject` is not a recommendation |

## Intake

The model converts the paragraph to JSON. Invalid JSON gets one repair call that includes the validation error. If the repair is still invalid, the chain stops with status `escalated`, decision `refer`, and a reason that names the parse failure.

The model also chooses `proceed`, `reject`, or `escalate` and gives a reason.

Required fields are full name, date of birth, vehicle year, make, model, and coverage. Occupation is optional. After the model returns, code checks the six required fields. If any is absent, null, or empty, the case cannot `proceed`, whatever the model said. A date of birth that is not `YYYY-MM-DD` counts as missing. When code overrides the model, the span records the override.

A missing full name or date of birth is `reject`. Status is `rejected`, the decision is `reject`, and the reason names the missing field, for example "required information date of birth is missing." Saying the fact will be called in later does not change this. If an identity field and a vehicle or coverage field are both missing, the case is still `reject`.

A missing vehicle year, make, model, or coverage, when full name and date of birth are present, is `escalate`. Status is `escalated`, the decision is `refer`, and the reason names the missing field.

A model `reject` on a complete application becomes `escalate`, recorded as an override.

Each of these stops ends the chain. Enrichment does not run.

## Enrichment

`src/underwriting/databases.py` does the lookups. The model does not search the files.

- Full match: one model call adds the bands, compares the applicant's claims with the records, and sets `discrepancy_flags` (a "clean record" claim when the history has accidents or tickets). The model returns one structured item per applicant statement with `applicant_said`, `record_shows`, and `contradicts`. Code keeps only items where `contradicts` is true and the quote appears in `applicant_claims`, then writes them as `discrepancy_flags`. Database rows are verified, so `verified` is always true. The prompt defines the scale. For claims, low means nothing at fault, and severe means two or more at-fault accidents or any bodily injury. Violations, the vehicle record, and the prior insurance band have their own steps on that same four-point scale. A missing record for a band is rated low, because the gap is already in `data_gaps`.
- Name matches and date of birth does not: stop with status `escalated`, decision `refer`, reason is a possible identity mismatch.
- No row in any database, or no driving-history row: stop with status `escalated`, decision `refer`, reason `insufficient information found`.
- Missing only from some databases, such as no prior-insurance row, while driving history exists: record `data_gaps` and continue.

An escalation does not call the later agents.

## Risk scoring and recommendation

Discrepancy flags and data gaps push the score up. They do not produce a confident low score.

Code sets the tier from the score with the ranges 0–29 low, 30–59 moderate, 60–79 high, and 80–100 severe. Code raises the score to at least 30 when there is any discrepancy flag or data gap, adding a floor factor.

The model decides approve, deny, or refer, using the seven rules in its prompt as guidance. Code guardrails only block an approve when there is a discrepancy flag or the tier is severe, and those cases become refer. An override triggers one extra model call for a rationale that matches the final decision, recorded as an override with the model's original decision and rationale. Conditions are kept only on approve: if that list is empty and data gaps are non-empty, one condition is added per gap in the form "resolve: <gap>". Deny and refer have no conditions. The model does the reasoning, and code enforces the lines that must never be crossed.

Only risk scoring retries, and only on timeout. A timeout in any other agent is not retried. It records one error span and escalates to refer. `MAX_ATTEMPTS` is 3. `BACKOFF_SECONDS` is `[1, 2]`: sleep 1 second after the first timeout and 2 seconds after the second. Tests pass a no-op sleep, so the suite does not wait. Each attempt is its own span with `attempt` 1, 2, or 3. A timed-out attempt has status `error`, `output` null, `error` set to the timeout message, and zero tokens unless that exception carries usage. If all three time out, `run_pipeline` returns status `escalated`, decision `refer`, and rationale `risk scoring timed out after 3 attempts; sent to human review`. That result has no score and no recommendation. The recommendation agent is not called. No score is invented, and the exception does not leave `run_pipeline`. A timeout does not also run a JSON repair.

The client timeout `DEFAULT_TIMEOUT_SECONDS` is 30 seconds, not 120. A warm local 9B call that is still running after 30 seconds is stuck. Three attempts at 120 seconds would hide that and would make the worst-case latency look like a batch job before the write-up measured it.

## Tracing and cost

One JSONL file per case: `traces/<case_id>.jsonl`. Each span has agent, attempt, start time, duration, prompt tokens, completion tokens, status, and any override or error. The runner prints a per-case summary table.

`case_id` is `person_id` after a name-and-DOB match. Before a match, it is the text-file stem.

Local Ollama charges $0 per token. The cost write-up still prices the measured token counts at one or two hosted models. Those rates stay placeholders until they are checked against the current pricing pages. Latency includes the worst case of three risk-scoring attempts plus backoff. The conclusion says whether that latency fits a real-time quote or an overnight batch.

## Cost and latency

`scripts/cost_report.py` reads `traces/<case_id>.jsonl`. Each span's `prompt_tokens` and `completion_tokens` are Ollama's `prompt_eval_count` and `eval_count`. A completed case has four spans and every status is `ok`. The averages use only those cases. Every other file is listed and left out.

<!-- paste cost_report output here -->

Local Ollama is $0 per token. Hosted prices stay unset on purpose. `PLACEHOLDER_RATES` in `src/underwriting/cost.py` names `hosted_model_A` and `hosted_model_B`, and both values are `None` until the input and output USD per 1M tokens are copied from that provider's current price page. `hosted_cost` is `(prompt_tokens / 1_000_000) * input + (completion_tokens / 1_000_000) * output`. A `None` rate prints "rate not set".

Worst-case latency, in milliseconds, is the normal case duration plus three client timeouts plus the backoff:

`normal_case_ms + timeouts * timeout_s * 1000 + backoff_s * 1000`

`timeouts` is 3 and `timeout_s` is 30 (`DEFAULT_TIMEOUT_SECONDS` in `src/underwriting/llm_adapter.py`). `backoff_s` is 3, the sum of `[1, 2]`. That sum is the path where risk scoring times out three times and the case escalates.

About 30 to 45 seconds per case on a local 9B model, plus 93 seconds when the three 30-second timeouts and the 3-second backoff fire, is too slow for a live quote. It fits batch processing, including an overnight batch, or an asynchronous "decision by email" flow. A faster hosted model, running intake and the database lookups in parallel, or caching repeated lookups is what would make a live quote realistic.

## Build plan

Rubric points are in brackets. Steps without a number support the pointed steps.

1. **Design doc and reads/produces table [8].** This file is that deliverable.
2. **Data.** Add the text files under `data/applications/`, the three databases under `data/databases/`, and point `data/manifest.json` at them. Remove the old JSON applications from the active set once the text files exist.
3. **Models.** Drop the `Raw*` application models. `IntakeRecord` keeps `raw_text` plus the normalized fields and the intake decision.
4. **Lookups.** Implement `src/underwriting/databases.py` and tests for a full match, a DOB mismatch, a missing driving-history row, and a gap in only one database.
5. **Fake adapter.** `tests/fakes.py` records calls, returns canned JSON, can return invalid JSON once, and can raise `LLMTimeoutError` a chosen number of times.
6. **Agents, one at a time [15].** Prompt and function, then the unit test, before the next agent. Each test uses a hardcoded input and the fake, checks the parsed model, checks that the prior message is still nested, and checks that the prompt received only that agent's slice. The prompt-injection text must not become an approve. All four unit tests pass before the chain calls an agent.
7. **Trace, then intake to enrichment [15].** Implement `tracing.py` (`record`, JSONL, short summaries). `run_pipeline` calls intake, applies the code floor, then enrichment. `tests/integration/test_handoff.py` checks the handoff and the two spans.
8. **Risk, recommendation, three real runs [15 + 8].** Add risk scoring, then recommendation, including the severe-tier and discrepancy guardrails. `tests/integration/test_full_chain.py` runs the fake-backed chain with no Ollama process. `scripts/run_cases.py` then runs the three cases that reach a final decision (clean, high-risk, mismatch) and writes their traces.
9. **Cost and latency [12].** Fill `docs/cost-and-latency.md` from those traces. State the model, the host, and the machine. State that local Ollama is $0 per token, price the same counts at the hosted placeholders, include worst-case retry latency, and judge real-time versus overnight batch.
10. **Retry, then escalate [17].** Lower the adapter timeout to 30 seconds. Retry risk scoring three times with a short backoff. `scripts/run_cases.py --inject-timeout` forces that path. `tests/integration/test_timeout_escalation.py` covers two timeouts then a success, and three timeouts then `escalated` / `refer` with three failed risk spans and no recommendation call. The pipeline does not raise.
11. **CI [10].** Add `httpx` as a direct dependency in `pyproject.toml`. Add `.github/workflows/ci.yml` to run pytest with the fake adapter and skip live-Ollama tests.
12. **README and screenshot checklist.** Document install, `ollama pull qwen3.5:9b`, pytest, and `python scripts/run_cases.py`. List the PDF screenshots in the assignment's order: design, each agent test, the handoff trace, the three full traces, the cost table, the timeout escalation, and the green CI run.

Scenarios to cover across the tests, with only the first three written out as end-to-end traces:

- Clean driver, reaches a decision
- High-risk driver whose text tells the model to approve, reaches a decision, must not approve
- Mismatch: claims a clean record, database has accidents, reaches a decision, discrepancy forces refer
- Not found in the databases, escalate
- Name match with the wrong date of birth, escalate
- Missing full name or date of birth, reject. `incomplete.txt` is missing the date of birth
- Missing vehicle year, make, model, or coverage while name and date of birth are present, escalate
- Messy formatting that intake can still normalize
- Timeout case, used by the retry test and `--inject-timeout`
