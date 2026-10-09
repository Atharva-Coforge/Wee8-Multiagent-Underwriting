"""Lookup rules against the three JSON databases."""

import json
from pathlib import Path

from underwriting.databases import lookup


def test_full_match_returns_all_three_records():
    result = lookup("Maria Elena Ortiz", "1984-05-12")

    assert result.match_status == "full_match"
    assert result.person_id == "P-1001"
    assert result.driving_history is not None
    assert result.driving_history.accidents == []
    assert result.driving_history.claims[0].type == "glass"
    assert result.vehicles is not None
    assert result.vehicles.vehicles[0].model == "CR-V"
    assert result.prior_insurance is not None
    assert result.prior_insurance.policies[0].lapse_days == 0


def test_dob_mismatch_when_name_matches_and_birth_date_does_not():
    result = lookup("Jordan A. Washington", "1990-03-14")

    assert result.match_status == "dob_mismatch"
    assert result.person_id == "P-1004"
    assert result.driving_history is not None
    assert result.driving_history.date_of_birth == "1988-03-14"
    assert result.vehicles is not None
    assert result.prior_insurance is not None


def test_missing_driving_history_row_is_not_found(tmp_path: Path):
    _write(
        tmp_path / "driving_history.json",
        [
            {
                "person_id": "P-1001",
                "full_name": "Maria Elena Ortiz",
                "date_of_birth": "1984-05-12",
                "accidents": [],
                "violations": [],
                "claims": [],
            }
        ],
    )
    _write(
        tmp_path / "vehicles_and_drivers.json",
        [
            {
                "person_id": "P-1008",
                "full_name": "Casey Nguyen",
                "date_of_birth": "1985-09-09",
                "vehicles": [
                    {
                        "year": 2016,
                        "make": "Ford",
                        "model": "Focus",
                        "currently_insured": True,
                    }
                ],
            }
        ],
    )
    _write(
        tmp_path / "prior_insurance.json",
        [
            {
                "person_id": "P-1008",
                "full_name": "Casey Nguyen",
                "date_of_birth": "1985-09-09",
                "policies": [
                    {
                        "carrier": "State Auto",
                        "currently_insured": True,
                        "years_continuous": 3,
                        "lapse_days": 0,
                        "cancelled": False,
                        "non_renewed": False,
                    }
                ],
            }
        ],
    )

    result = lookup("Casey Nguyen", "1985-09-09", database_dir=tmp_path)

    assert result.match_status == "not_found"
    assert result.person_id == "P-1008"
    assert result.driving_history is None
    assert result.vehicles is not None
    assert result.prior_insurance is not None


def test_gap_in_only_prior_insurance_still_matches():
    result = lookup("Robin Hale", "1980-01-15")

    assert result.match_status == "full_match"
    assert result.person_id == "P-1007"
    assert result.driving_history is not None
    assert result.vehicles is not None
    assert result.prior_insurance is None


def test_unknown_person_is_not_found():
    result = lookup("Samir Cole", "1995-06-01")

    assert result.match_status == "not_found"
    assert result.person_id is None
    assert result.driving_history is None
    assert result.vehicles is None
    assert result.prior_insurance is None


def test_two_digit_year_in_the_future_rolls_back_a_century(tmp_path: Path):
    _write(
        tmp_path / "driving_history.json",
        [
            {
                "person_id": "P-1960",
                "full_name": "Pat Ellis",
                "date_of_birth": "1960-03-22",
                "accidents": [],
                "violations": [],
                "claims": [],
            }
        ],
    )
    _write(tmp_path / "vehicles_and_drivers.json", [])
    _write(tmp_path / "prior_insurance.json", [])

    result = lookup("Pat Ellis", "03/22/60", database_dir=tmp_path)

    assert result.match_status == "full_match"
    assert result.person_id == "P-1960"
    assert result.driving_history is not None
    assert result.driving_history.date_of_birth == "1960-03-22"


def test_blank_name_or_unreadable_date_is_not_found():
    assert lookup("", "1984-05-12").match_status == "not_found"
    assert lookup("Maria Elena Ortiz", "not a date").match_status == "not_found"


def test_normalized_name_and_date_match():
    result = lookup("RIVERA, ALEX M.", "03/22/91")

    assert result.match_status == "full_match"
    assert result.person_id == "P-1005"
    assert result.driving_history is not None
    assert result.driving_history.full_name == "Alex M. Rivera"
    assert result.driving_history.date_of_birth == "1991-03-22"


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"records": records}))
