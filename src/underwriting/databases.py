"""Look up an applicant in the three JSON databases."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from underwriting.models import (
    DrivingHistoryRecord,
    LookupResult,
    PriorInsuranceRecord,
    VehicleRecord,
)

T = TypeVar("T", bound=BaseModel)

_Record = DrivingHistoryRecord | VehicleRecord | PriorInsuranceRecord


def lookup(
    full_name: str,
    date_of_birth: str,
    *,
    database_dir: Path | None = None,
) -> LookupResult:
    """Match a normalized name and date of birth.

    ``full_match`` requires a driving-history row. A name match with a different
    date of birth is ``dob_mismatch``. No name match, or a name-and-date match
    with no driving-history row, is ``not_found``.
    """

    root = database_dir or _default_database_dir()
    driving = _load(root / "driving_history.json", DrivingHistoryRecord)
    vehicles = _load(root / "vehicles_and_drivers.json", VehicleRecord)
    policies = _load(root / "prior_insurance.json", PriorInsuranceRecord)

    name_key = _normalize_name(full_name)
    dob_key = _normalize_date(date_of_birth)
    if not name_key or dob_key is None:
        return LookupResult(match_status="not_found")

    same_driving = _pick(driving, name_key, dob_key, same_dob=True)
    same_vehicles = _pick(vehicles, name_key, dob_key, same_dob=True)
    same_policies = _pick(policies, name_key, dob_key, same_dob=True)
    if same_driving is not None:
        return LookupResult(
            match_status="full_match",
            person_id=same_driving.person_id,
            driving_history=same_driving,
            vehicles=same_vehicles,
            prior_insurance=same_policies,
        )
    if same_vehicles is not None or same_policies is not None:
        return LookupResult(
            match_status="not_found",
            person_id=_person_id(same_vehicles, same_policies),
            vehicles=same_vehicles,
            prior_insurance=same_policies,
        )

    other_driving = _pick(driving, name_key, dob_key, same_dob=False)
    other_vehicles = _pick(vehicles, name_key, dob_key, same_dob=False)
    other_policies = _pick(policies, name_key, dob_key, same_dob=False)
    if other_driving or other_vehicles or other_policies:
        return LookupResult(
            match_status="dob_mismatch",
            person_id=_person_id(other_driving, other_vehicles, other_policies),
            driving_history=other_driving,
            vehicles=other_vehicles,
            prior_insurance=other_policies,
        )
    return LookupResult(match_status="not_found")


def _default_database_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "databases"


def _load(path: Path, model: type[T]) -> list[T]:
    payload = json.loads(path.read_text())
    return [model.model_validate(row) for row in payload["records"]]


def _pick(rows: list[T], name_key: str, dob_key: str, *, same_dob: bool) -> T | None:
    for row in rows:
        if _normalize_name(row.full_name) != name_key:
            continue
        matches_dob = _normalize_date(row.date_of_birth) == dob_key
        if matches_dob == same_dob:
            return row
    return None


def _person_id(*rows: _Record | None) -> str | None:
    for row in rows:
        if row is not None:
            return row.person_id
    return None


def _normalize_name(name: str) -> str:
    text = " ".join(name.replace(".", " ").split())
    if "," in text:
        last, _, rest = text.partition(",")
        text = " ".join(f"{rest} {last}".split())
    return text.casefold()


def _normalize_date(value: str) -> str | None:
    """ISO and written dates. A two-digit year that lands in the future rolls back one century."""

    text = " ".join(value.strip().split())
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.strptime(text, "%m/%d/%y").date()
    except ValueError:
        return None
    if parsed > date.today():
        try:
            parsed = parsed.replace(year=parsed.year - 100)
        except ValueError:
            return None
    return parsed.isoformat()
