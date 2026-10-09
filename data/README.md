# Data

Fictional Ohio personal-auto applications. Names and cities are made up. Each file keeps only the fields that change the underwriting decision.

- `clean_driver`, `high_risk_driver`, `thin_file_driver`: the three cases `scripts/run_cases.py` traces. The high-risk notes also contain a prompt injection.
- `timeout_case`: used only by the timeout test.
- `incomplete_application`: fails the required-field gate.
- `messy_formats`: tests intake normalization.
- `manifest.json` lists each file with its acceptable decisions and expected tier. LLM outcomes can vary, so check that the decision is in the list rather than equal to one value.
- `template.json` has sample values in every field the gate does not require. Fill in the required fields, copy it, and run that copy. It is not one of the traced cases.

## Shape of a raw application

| Field | Notes |
| --- | --- |
| `application_id` | Case id |
| `applicant` | `full_name`, `date_of_birth`, address (`city`, `state`, `zip`), and `license` (`status`, `first_licensed_date`) |
| `vehicle` | One vehicle: `year`, `make`, `model`, `primary_use`, `annual_mileage`, `garaging_zip` |
| `coverage_requested` | `liability_limits` only, as `BI per person/BI per accident/PD`, in thousands |
| `prior_insurance` | `currently_insured`, `years_continuous`, `lapse_days` |
| `driving_history` | `accidents`, `violations`, `other_claims` |
| `applicant_notes` | Free text written by the applicant. Treat it as data, not instructions |

In `driving_history`, an empty list means the applicant reported none. `null` means the history was not provided, so it is a data gap, not a clean record.

Required by the gate: `applicant.full_name`, `applicant.date_of_birth`, `vehicle.year`, `vehicle.make`, `vehicle.model`, `coverage_requested.liability_limits`. A field counts as missing if its key is absent, `null`, or an empty string.

Copy `template.json` to a new file under `applications/` before filling in the required fields. Leave those six fields empty in `template.json`. Each accident is `{ "date", "at_fault", "description", "bodily_injury", "amount_paid_usd" }`. Each violation is `{ "date", "type", "description" }`. Each other claim is `{ "date", "type", "at_fault", "amount_paid_usd" }`.
