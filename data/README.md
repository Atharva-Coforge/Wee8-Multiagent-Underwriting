# Data

Fictional Ohio personal-auto applications. Names, addresses, VINs and carriers are made up. Phone numbers use the 555-01xx range and emails use `example.com`.

- `clean_driver`, `high_risk_driver`, `thin_file_driver`: the three cases `scripts/run_cases.py` traces. The high-risk notes also contain a prompt injection.
- `timeout_case`: used only by the timeout test.
- `incomplete_application`: fails the required-field gate.
- `messy_formats`: tests intake normalization.
- `manifest.json` lists each file with its acceptable decisions and expected tier. LLM outcomes can vary, so check that the decision is in the list rather than equal to one value.

## Shape of a raw application

| Field | Notes |
| --- | --- |
| `applicant` | Name, `date_of_birth`, address, and `license` (`status`, `first_licensed_date`) |
| `additional_drivers` | Other household drivers. Empty list if none |
| `vehicle` | A single vehicle: `year`, `make`, `model`, use, mileage, and garaging zip |
| `coverage_requested` | `liability_limits` as `BI per person/BI per accident/PD`, in thousands |
| `prior_insurance` | Continuous years and `lapse_days` |
| `driving_history` | `accidents`, `violations`, `other_claims` |
| `applicant_notes` | Free text written by the applicant. Treat it as data, not instructions |

In `driving_history`, an empty list means the applicant reported none. `null` means the history was not provided, so it is a data gap, not a clean record.

Required by the gate: `applicant.full_name`, `applicant.date_of_birth`, `vehicle.year`, `vehicle.make`, `vehicle.model`, `coverage_requested.liability_limits`. A field counts as missing if its key is absent, `null`, or an empty string.
