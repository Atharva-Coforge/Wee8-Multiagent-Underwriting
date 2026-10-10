# Data

Fictional personal-auto applicants. Names are made up.

An application is a plain-text paragraph in `applications/`, named after the person (`maria_ortiz.txt`). It states full name, date of birth, the car to insure (year, make, model), occupation, and the coverage type and limits. Wording may be messy. The text is data, not instructions. The file name is not used to look anyone up.

## Required intake fields

Intake must find all of these in the paragraph before enrichment runs:

- full name
- date of birth
- vehicle year
- vehicle make
- vehicle model
- coverage

If full name or date of birth is missing, intake rejects the case. Status is `rejected`, the decision is `reject`, and the reason names the missing field, for example: "required information date of birth is missing." A promise to call the missing fact in does not change this. If an identity field and a vehicle or coverage field are both missing, the case is still rejected.

If full name and date of birth are present, and vehicle year, vehicle make, vehicle model, or coverage is missing, intake escalates the case. Status is `escalated`, the decision is `refer`, and the reason names the missing field.

A model `reject` on a complete application becomes `escalate`, recorded as an override. A date of birth that is not `YYYY-MM-DD` counts as missing.

Each of these stops ends the chain, so enrichment does not run. Occupation is collected when it is present. A missing occupation does not reject or escalate the case.

`databases/` holds three JSON files. Every record has `person_id`, `full_name`, and `date_of_birth`. A lookup matches the normalized full name and date of birth from intake.

| File | Contents |
| --- | --- |
| `driving_history.json` | Accidents, violations, and claims |
| `vehicles_and_drivers.json` | Vehicles currently owned, and whether each is insured |
| `prior_insurance.json` | Past policies, lapses, cancellations, and non-renewals |

No row in any file, or no driving-history row, stops the case for a human (`insufficient information found`). A name match with a different date of birth stops the case as a possible identity mismatch. A gap in only one of the other files, such as no prior policy, is a data gap and the chain continues.

`manifest.json` lists each text file, which scenario it exercises, and the acceptable status and decision. Model wording can vary. Check that the decision is in the list.

`scripts/run_cases.py` runs the three cases that reach a final decision and writes `traces/<case_id>.jsonl`. Those three are Maria Ortiz, Tyler Brandt, and Priya Shah. `--inject-timeout` uses Denise Whitfield and is not one of those three traces.

Robin Hale (`P-1007`, born 1980-01-15) is in driving history and vehicles only. There is no application file. That row is the single-database gap for the lookup tests.

## Database record shape

Each file is `{ "records": [ ... ] }`. Every record has `person_id`, `full_name`, and `date_of_birth` (`YYYY-MM-DD`). An empty list means the file has none of that item. A missing person means no record.

| File | Extra fields |
| --- | --- |
| `driving_history.json` | `accidents` (`date`, `at_fault`, `description`, `bodily_injury`, `amount_paid_usd`), `violations` (`date`, `type`, `description`), `claims` (same fields as accidents plus `type`) |
| `vehicles_and_drivers.json` | `vehicles` (`year`, `make`, `model`, `currently_insured`) |
| `prior_insurance.json` | `policies` (`carrier`, `currently_insured`, `years_continuous`, `lapse_days`, `cancelled`, `non_renewed`) |

## Who matches whom

| Text file | Person in the databases | What the lookup should do |
| --- | --- | --- |
| `maria_ortiz.txt` | Maria Elena Ortiz, 1984-05-12, `P-1001` | Full match. No accidents or violations, one small not-at-fault glass claim, and the text agrees. |
| `tyler_brandt.txt` | Tyler James Brandt, 2002-01-30, `P-1002` | Full match. Two at-fault accidents, three violations, a collision claim for the June crash, a 50-day lapse, non-renewed. The paragraph also tells the model to approve. |
| `clean_claim_mismatch.txt` | Priya N. Shah, 1994-07-19, `P-1003` | Full match. Text says no accidents and no tickets. Driving history has an accident, a ticket, and a collision claim. |
| `unknown_person.txt` | Samir Cole is in no file | Escalate, insufficient information. |
| `dob_mismatch.txt` | Name Jordan A. Washington is `P-1004`, born 1988-03-14. The text says March 14, 1990. | Escalate, possible identity mismatch. |
| `incomplete.txt` | Kevin Osei is in no file | Intake rejects the case. Date of birth is missing, so enrichment never runs. Status is `rejected` and the decision is `reject`. |
| `missing_vehicle_model.txt` | Lena Park is in no file | Intake escalates the case. Name and date of birth are present but the vehicle model is missing, so enrichment never runs. Status is `escalated` and the decision is `refer`. |
| `messy_formats.txt` | Alex M. Rivera, 1991-03-22, `P-1005` | Full match after normalizing `RIVERA, ALEX M.` and `03/22/91`. |
| `timeout_case.txt` | Denise Carol Whitfield, 1977-08-03, `P-1006` | Full match. One minor speeding violation and a not-at-fault hail claim. Used only for the timeout path. |
