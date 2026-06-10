from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLAIMS_PATH = PROJECT_ROOT / "data" / "sample" / "sample_claims.csv"
DEFAULT_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "drug_master.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_demand.parquet"

CLAIM_YEAR_COLUMN = "費用年"
CLAIM_DRUG_CODE_COLUMN = "藥品代碼"
CLAIM_QTY_COLUMN_CANDIDATES = [
    "含包裹支付的醫令量_合計",
    "醫令量_合計",
]

OUTPUT_COLUMNS = [
    "shortage_group_key",
    "total_claim_qty_latest_year",
    "demand_rank_percentile",
    "demand_share_within_group",
    "claim_qty_yoy_growth",
    "claim_qty_3y_cv",
]


@dataclass(frozen=True)
class BuildDemandFeaturesResult:
    output_path: Path
    schema: list[str]
    row_count: int
    latest_year: int | None
    available_years: list[int]
    assumptions: list[str]


def find_claim_qty_column(claims: pd.DataFrame) -> str:
    for column in CLAIM_QTY_COLUMN_CANDIDATES:
        if column in claims.columns:
            return column
    candidates = ", ".join(CLAIM_QTY_COLUMN_CANDIDATES)
    raise ValueError(f"claims data must include one of: {candidates}.")


def read_claims(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def prepare_claims(claims: pd.DataFrame) -> pd.DataFrame:
    if CLAIM_YEAR_COLUMN not in claims.columns:
        raise ValueError(f"claims data must include {CLAIM_YEAR_COLUMN}.")
    if CLAIM_DRUG_CODE_COLUMN not in claims.columns:
        raise ValueError(f"claims data must include {CLAIM_DRUG_CODE_COLUMN}.")

    qty_column = find_claim_qty_column(claims)
    frame = claims[[CLAIM_YEAR_COLUMN, CLAIM_DRUG_CODE_COLUMN, qty_column]].copy()
    frame = frame.rename(
        columns={
            CLAIM_YEAR_COLUMN: "claim_year",
            CLAIM_DRUG_CODE_COLUMN: "drug_code",
            qty_column: "claim_qty",
        }
    )
    frame["claim_year"] = pd.to_numeric(frame["claim_year"], errors="coerce")
    frame["claim_qty"] = pd.to_numeric(frame["claim_qty"], errors="coerce").fillna(0.0)
    frame["drug_code"] = frame["drug_code"].astype("string").str.strip()
    return frame.dropna(subset=["claim_year", "drug_code"])


def calculate_latest_year_features(joined: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    latest = joined.loc[joined["claim_year"] == latest_year].copy()
    group_totals = (
        latest.groupby("shortage_group_key", dropna=False, as_index=False)["claim_qty"]
        .sum()
        .rename(columns={"claim_qty": "total_claim_qty_latest_year"})
    )
    drug_totals = (
        latest.groupby(["shortage_group_key", "drug_code"], dropna=False, as_index=False)[
            "claim_qty"
        ]
        .sum()
        .rename(columns={"claim_qty": "drug_claim_qty_latest_year"})
    )
    drug_totals = drug_totals.merge(group_totals, on="shortage_group_key", how="left")
    drug_totals["drug_share_within_group"] = (
        drug_totals["drug_claim_qty_latest_year"]
        / drug_totals["total_claim_qty_latest_year"].where(
            drug_totals["total_claim_qty_latest_year"] != 0
        )
    ).fillna(0.0)
    max_share = (
        drug_totals.groupby("shortage_group_key", dropna=False, as_index=False)[
            "drug_share_within_group"
        ]
        .max()
        .rename(columns={"drug_share_within_group": "demand_share_within_group"})
    )
    features = group_totals.merge(max_share, on="shortage_group_key", how="left")
    features["demand_rank_percentile"] = features["total_claim_qty_latest_year"].rank(pct=True)
    return features


def calculate_multi_year_features(joined: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    yearly = (
        joined.groupby(["shortage_group_key", "claim_year"], dropna=False, as_index=False)[
            "claim_qty"
        ]
        .sum()
        .rename(columns={"claim_qty": "annual_claim_qty"})
    )
    previous_year = latest_year - 1
    latest = yearly.loc[
        yearly["claim_year"] == latest_year,
        ["shortage_group_key", "annual_claim_qty"],
    ]
    previous = yearly.loc[
        yearly["claim_year"] == previous_year, ["shortage_group_key", "annual_claim_qty"]
    ]
    growth = latest.merge(
        previous,
        on="shortage_group_key",
        how="left",
        suffixes=("_latest", "_prev"),
    )
    growth["claim_qty_yoy_growth"] = (
        (growth["annual_claim_qty_latest"] - growth["annual_claim_qty_prev"])
        / growth["annual_claim_qty_prev"].where(growth["annual_claim_qty_prev"] != 0)
    )
    growth = growth[["shortage_group_key", "claim_qty_yoy_growth"]]

    recent_years = [latest_year - 2, previous_year, latest_year]
    three_year = yearly.loc[yearly["claim_year"].isin(recent_years)]
    cv = (
        three_year.groupby("shortage_group_key", dropna=False)["annual_claim_qty"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    cv["claim_qty_3y_cv"] = cv["std"] / cv["mean"].where((cv["count"] >= 3) & (cv["mean"] != 0))
    return growth.merge(cv[["shortage_group_key", "claim_qty_3y_cv"]], on="shortage_group_key")


def build_demand_features(
    claims_path: Path = DEFAULT_CLAIMS_PATH,
    master_path: Path = DEFAULT_MASTER_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> BuildDemandFeaturesResult:
    claims = prepare_claims(read_claims(claims_path))
    master = pd.read_parquet(master_path)
    required_master_columns = {"drug_code", "shortage_group_key"}
    missing_master_columns = required_master_columns - set(master.columns)
    if missing_master_columns:
        missing = ", ".join(sorted(missing_master_columns))
        raise ValueError(f"drug master missing required columns: {missing}.")

    mapping = master[["drug_code", "shortage_group_key"]].drop_duplicates()
    joined = claims.merge(mapping, on="drug_code", how="inner")
    available_years = sorted(int(year) for year in joined["claim_year"].dropna().unique())
    latest_year = max(available_years) if available_years else None

    if latest_year is None:
        features = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        features = calculate_latest_year_features(joined, latest_year)
        if len(available_years) > 1:
            multi_year = calculate_multi_year_features(joined, latest_year)
            features = features.merge(multi_year, on="shortage_group_key", how="left")
        else:
            features["claim_qty_yoy_growth"] = pd.NA
            features["claim_qty_3y_cv"] = pd.NA
        features = features[OUTPUT_COLUMNS].sort_values("shortage_group_key").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    if available_years == [113]:
        year_assumption = (
            "Only 113-year claims data is available; YoY growth and 3-year CV "
            "cannot be calculated yet."
        )
    elif len(available_years) <= 1:
        year_assumption = (
            "Only one claims year is available; YoY growth and 3-year CV cannot "
            "be calculated yet."
        )
    else:
        year_assumption = (
            "YoY growth uses latest year versus previous year; 3-year CV uses the "
            "latest three calendar years when present."
        )

    return BuildDemandFeaturesResult(
        output_path=output_path,
        schema=OUTPUT_COLUMNS,
        row_count=len(features),
        latest_year=latest_year,
        available_years=available_years,
        assumptions=[
            "Claims are joined to drug_master by drug_code and aggregated by shortage_group_key.",
            (
                "total_claim_qty_latest_year uses the latest available claims year "
                "after joining to drug_master."
            ),
            (
                "demand_rank_percentile ranks group demand across shortage groups; "
                "larger demand has a higher percentile."
            ),
            (
                "demand_share_within_group is the largest single drug_code share "
                "inside the shortage group."
            ),
            year_assumption,
        ],
    )


def print_summary(result: BuildDemandFeaturesResult) -> None:
    print(f"Wrote {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Latest year: {result.latest_year}")
    print(f"Available years: {result.available_years}")
    print("Schema:")
    for column in result.schema:
        print(f"- {column}")
    print("Assumptions:")
    for assumption in result.assumptions:
        print(f"- {assumption}")


def main() -> int:
    result = build_demand_features()
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
