from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "drug_master.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_supply.parquet"
EXPIRING_WITHIN_DAYS = 180

OUTPUT_COLUMNS = [
    "shortage_group_key",
    "active_license_count",
    "manufacturer_count",
    "import_manufacturer_ratio",
    "single_supplier_flag",
    "expiring_license_count",
]

EXPIRY_COLUMN_CANDIDATES = [
    "license_expiry_date",
    "expiry_date",
    "valid_until",
    "effective_until",
]

DOMESTIC_COUNTRY_VALUES = {"TW", "TAIWAN", "ROC", "R.O.C.", "台灣", "臺灣"}


@dataclass(frozen=True)
class BuildSupplyFeaturesResult:
    output_path: Path
    schema: list[str]
    row_count: int
    assumptions: list[str]


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text.upper() if text else None


def find_expiry_column(frame: pd.DataFrame) -> str | None:
    for column in EXPIRY_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def parse_expiry_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(values, format="%Y/%m/%d", errors="coerce")


def add_license_activity_columns(master: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    frame = master.copy()
    expiry_column = find_expiry_column(frame)
    frame["_license_no_normalized"] = frame["license_no"].map(normalize_text)
    frame["_has_license"] = frame["_license_no_normalized"].notna()
    if expiry_column is None:
        frame["_license_expiry_date"] = pd.NaT
        frame["_active_license"] = frame["_has_license"]
        frame["_expiring_license"] = False
        return frame

    expiry_dates = parse_expiry_dates(frame[expiry_column])
    as_of_timestamp = pd.Timestamp(as_of_date)
    expiry_window_end = as_of_timestamp + pd.Timedelta(days=EXPIRING_WITHIN_DAYS)
    frame["_license_expiry_date"] = expiry_dates
    frame["_active_license"] = frame["_has_license"] & (
        expiry_dates.isna() | (expiry_dates >= as_of_timestamp)
    )
    frame["_expiring_license"] = (
        frame["_has_license"]
        & expiry_dates.notna()
        & (expiry_dates >= as_of_timestamp)
        & (expiry_dates <= expiry_window_end)
    )
    return frame


def calculate_group_features(group: pd.DataFrame) -> pd.Series:
    active_license_count = group.loc[
        group["_active_license"], "_license_no_normalized"
    ].nunique()
    manufacturer_country = (
        group[["manufacturer", "country"]]
        .dropna(subset=["manufacturer"])
        .assign(
            manufacturer_normalized=lambda frame: frame["manufacturer"].map(normalize_text),
            country_normalized=lambda frame: frame["country"].map(normalize_text),
        )
        .drop_duplicates(subset=["manufacturer_normalized"])
    )
    manufacturer_count = manufacturer_country["manufacturer_normalized"].nunique()
    imported_manufacturer_count = manufacturer_country.loc[
        manufacturer_country["country_normalized"].notna()
        & ~manufacturer_country["country_normalized"].isin(DOMESTIC_COUNTRY_VALUES),
        "manufacturer_normalized",
    ].nunique()
    expiring_license_count = group.loc[
        group["_expiring_license"], "_license_no_normalized"
    ].nunique()

    return pd.Series(
        {
            "active_license_count": int(active_license_count),
            "manufacturer_count": int(manufacturer_count),
            "import_manufacturer_ratio": (
                imported_manufacturer_count / manufacturer_count if manufacturer_count else 0.0
            ),
            "single_supplier_flag": bool(manufacturer_count == 1),
            "expiring_license_count": int(expiring_license_count),
        }
    )


def build_supply_features(
    master_path: Path = DEFAULT_MASTER_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    as_of_date: date | None = None,
) -> BuildSupplyFeaturesResult:
    if as_of_date is None:
        as_of_date = date.today()
    master = pd.read_parquet(master_path)
    if "shortage_group_key" not in master.columns:
        raise ValueError("drug master must include shortage_group_key.")
    if "license_no" not in master.columns:
        raise ValueError("drug master must include license_no.")

    frame = add_license_activity_columns(master, as_of_date)
    features = (
        frame.groupby("shortage_group_key", dropna=False)
        .apply(calculate_group_features, include_groups=False)
        .reset_index()
    )
    features = features[OUTPUT_COLUMNS].sort_values("shortage_group_key").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    expiry_column = find_expiry_column(master)
    expiry_assumption = (
        f"Expiring licenses are counted from `{expiry_column}` within "
        f"{EXPIRING_WITHIN_DAYS} days of {as_of_date.isoformat()}."
        if expiry_column
        else "No license expiry column was available; expiring_license_count is 0."
    )
    return BuildSupplyFeaturesResult(
        output_path=output_path,
        schema=OUTPUT_COLUMNS,
        row_count=len(features),
        assumptions=[
            "Features are calculated per shortage_group_key.",
            (
                "active_license_count counts distinct non-null licenses unless an "
                "expiry column marks them expired."
            ),
            "manufacturer_count counts distinct non-null manufacturers.",
            (
                "import_manufacturer_ratio treats country values other than "
                "TW/Taiwan variants as imported."
            ),
            "single_supplier_flag is true when exactly one manufacturer is present in the group.",
            expiry_assumption,
        ],
    )


def print_summary(result: BuildSupplyFeaturesResult) -> None:
    print(f"Wrote {result.output_path}")
    print(f"Rows: {result.row_count}")
    print("Schema:")
    for column in result.schema:
        print(f"- {column}")
    print("Assumptions:")
    for assumption in result.assumptions:
        print(f"- {assumption}")


def main() -> int:
    result = build_supply_features()
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
