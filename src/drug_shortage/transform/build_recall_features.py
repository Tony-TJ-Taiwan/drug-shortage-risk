from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECALLS_PATH = PROJECT_ROOT / "data" / "sample" / "sample_recalls.csv"
DEFAULT_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "drug_master.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_recall.parquet"

RECALL_LEVEL_COLUMN = "回收分級"
RECALL_DATE_COLUMN = "日期"
RECALL_PRODUCT_COLUMN = "產品"
RECALL_LICENSE_COLUMN = "許可證字號"

OUTPUT_COLUMNS = [
    "shortage_group_key",
    "recall_count_12m",
    "recall_count_24m",
    "severe_recall_flag",
    "latest_recall_date",
    "match_confidence",
]


@dataclass(frozen=True)
class BuildRecallFeaturesResult:
    output_path: Path
    schema: list[str]
    row_count: int
    matched_recall_count: int
    assumptions: list[str]


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace('"', "").replace("“", "").replace("”", "")
    return re.sub(r"\s+", "", text).upper()


def license_keys_from_text(value: object) -> list[str]:
    text = str(value).strip() if not pd.isna(value) else ""
    if not text:
        return []
    keys = []
    for part in re.split(r"[、,;；\s]+", text):
        digits = "".join(re.findall(r"\d+", part))
        if len(digits) >= 5:
            keys.append(digits[-5:])
    return keys


def parse_recall_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def is_severe_recall(value: object) -> bool:
    text = normalize_text(value)
    if text is None:
        return False
    return text.startswith("1") or "第一級" in text


def prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"shortage_group_key", "license_no", "product_name"}
    missing_columns = required_columns - set(master.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"drug master missing required columns: {missing}.")

    frame = master[["shortage_group_key", "license_no", "product_name"]].copy()
    frame["_license_key"] = frame["license_no"].map(
        lambda value: license_keys_from_text(value)[0] if license_keys_from_text(value) else None
    )
    frame["_product_name_normalized"] = frame["product_name"].map(normalize_text)
    return frame


def prepare_recalls(recalls: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        RECALL_LEVEL_COLUMN,
        RECALL_DATE_COLUMN,
        RECALL_PRODUCT_COLUMN,
        RECALL_LICENSE_COLUMN,
    }
    missing_columns = required_columns - set(recalls.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"recalls data missing required columns: {missing}.")

    frame = recalls[
        [RECALL_LEVEL_COLUMN, RECALL_DATE_COLUMN, RECALL_PRODUCT_COLUMN, RECALL_LICENSE_COLUMN]
    ].copy()
    frame = frame.rename(
        columns={
            RECALL_LEVEL_COLUMN: "recall_level",
            RECALL_DATE_COLUMN: "recall_date",
            RECALL_PRODUCT_COLUMN: "product_name",
            RECALL_LICENSE_COLUMN: "license_no",
        }
    )
    frame["recall_date"] = parse_recall_dates(frame["recall_date"])
    frame["severe_recall"] = frame["recall_level"].map(is_severe_recall)
    frame["_recall_id"] = range(len(frame))
    frame["_license_keys"] = frame["license_no"].map(license_keys_from_text)
    frame["_product_name_normalized"] = frame["product_name"].map(normalize_text)
    return frame


def match_recalls_to_master(recalls: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    license_recalls = recalls.loc[recalls["_license_keys"].map(bool)].explode("_license_keys")
    license_matches = license_recalls.merge(
        master,
        left_on="_license_keys",
        right_on="_license_key",
        how="inner",
        suffixes=("_recall", "_master"),
    )
    license_matches["match_confidence"] = "high"

    product_recalls = recalls.loc[~recalls["_license_keys"].map(bool)]
    product_matches = product_recalls.merge(
        master,
        on="_product_name_normalized",
        how="inner",
        suffixes=("_recall", "_master"),
    )
    product_matches["match_confidence"] = "low"

    matches = pd.concat([license_matches, product_matches], ignore_index=True, sort=False)
    if matches.empty:
        return matches
    return matches.drop_duplicates(
        subset=["_recall_id", "shortage_group_key", "match_confidence"]
    )


def summarize_group_recalls(matches: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "shortage_group_key",
                "recall_count_12m",
                "recall_count_24m",
                "severe_recall_flag",
                "latest_recall_date",
                "match_confidence",
            ]
        )

    frame = matches.copy()
    window_12m_start = as_of_date - pd.DateOffset(months=12)
    window_24m_start = as_of_date - pd.DateOffset(months=24)
    frame["_in_12m"] = frame["recall_date"].between(window_12m_start, as_of_date, inclusive="both")
    frame["_in_24m"] = frame["recall_date"].between(window_24m_start, as_of_date, inclusive="both")

    grouped = frame.groupby("shortage_group_key", dropna=False)
    features = grouped.agg(
        recall_count_12m=("_in_12m", "sum"),
        recall_count_24m=("_in_24m", "sum"),
        severe_recall_flag=("severe_recall", "max"),
        latest_recall_date=("recall_date", "max"),
    ).reset_index()
    low_only = grouped["match_confidence"].agg(
        lambda values: "high" if "high" in set(values) else "low"
    )
    features = features.merge(low_only.rename("match_confidence"), on="shortage_group_key")
    features["recall_count_12m"] = features["recall_count_12m"].astype(int)
    features["recall_count_24m"] = features["recall_count_24m"].astype(int)
    features["severe_recall_flag"] = features["severe_recall_flag"].astype(bool)
    return features


def build_recall_features(
    recalls_path: Path = DEFAULT_RECALLS_PATH,
    master_path: Path = DEFAULT_MASTER_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    as_of_date: pd.Timestamp | None = None,
) -> BuildRecallFeaturesResult:
    recalls = prepare_recalls(pd.read_csv(recalls_path, dtype=str, keep_default_na=False))
    master = prepare_master(pd.read_parquet(master_path))
    matches = match_recalls_to_master(recalls, master)
    if as_of_date is None:
        as_of_date = matches["recall_date"].max() if not matches.empty else pd.Timestamp.today()
    as_of_date = pd.Timestamp(as_of_date).normalize()

    matched_features = summarize_group_recalls(matches, as_of_date)
    all_groups = master[["shortage_group_key"]].drop_duplicates()
    features = all_groups.merge(matched_features, on="shortage_group_key", how="left")
    features["recall_count_12m"] = features["recall_count_12m"].fillna(0).astype(int)
    features["recall_count_24m"] = features["recall_count_24m"].fillna(0).astype(int)
    features["severe_recall_flag"] = features["severe_recall_flag"].fillna(False).astype(bool)
    features["match_confidence"] = features["match_confidence"].fillna("none")
    features = features[OUTPUT_COLUMNS].sort_values("shortage_group_key").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return BuildRecallFeaturesResult(
        output_path=output_path,
        schema=OUTPUT_COLUMNS,
        row_count=len(features),
        matched_recall_count=matches["_recall_id"].nunique() if not matches.empty else 0,
        assumptions=[
            "Recall features are calculated per shortage_group_key.",
            "Recall rows are matched by license number first.",
            "Product name matching is used only when a recall row has no license number.",
            "Product-name-only matches are marked with low match_confidence.",
            f"12-month and 24-month recall windows are calculated as of {as_of_date.date()}.",
            "severe_recall_flag is true for first-class recall levels.",
        ],
    )


def print_summary(result: BuildRecallFeaturesResult) -> None:
    print(f"Wrote {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Matched recalls: {result.matched_recall_count}")
    print("Schema:")
    for column in result.schema:
        print(f"- {column}")
    print("Assumptions:")
    for assumption in result.assumptions:
        print(f"- {assumption}")


def main() -> int:
    result = build_recall_features()
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
