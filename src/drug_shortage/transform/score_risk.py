from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "drug_master.parquet"
DEFAULT_SUPPLY_PATH = PROJECT_ROOT / "data" / "processed" / "features_supply.parquet"
DEFAULT_DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "features_demand.parquet"
DEFAULT_RECALL_PATH = PROJECT_ROOT / "data" / "processed" / "features_recall.parquet"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "scoring.yml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "shortage_risk_scores.parquet"

OUTPUT_COLUMNS = [
    "shortage_group_key",
    "supply_score",
    "demand_score",
    "recall_score",
    "global_shortage_score",
    "price_or_exit_score",
    "total_risk_score",
    "risk_level",
    "risk_explanation",
]

COMPONENT_COLUMNS = [
    "supply_score",
    "demand_score",
    "recall_score",
    "global_shortage_score",
    "price_or_exit_score",
]
OPTIONAL_COMPONENT_COLUMNS = ["global_shortage_score", "price_or_exit_score"]


@dataclass(frozen=True)
class ScoreRiskResult:
    output_path: Path
    schema: list[str]
    row_count: int
    assumptions: list[str]


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    if pd.isna(value):
        return minimum
    return max(minimum, min(maximum, float(value)))


def bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if "weights" not in config:
        raise ValueError("scoring config must define weights.")
    return config


def safe_read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def score_supply(row: pd.Series, rules: dict[str, Any]) -> float:
    if not bool_value(row.get("_supply_available")):
        return 0.0
    license_rules = rules.get("active_license_count", {})
    active_license_count = row.get("active_license_count")
    if pd.isna(active_license_count) or active_license_count <= 0:
        score = license_rules.get("missing_or_zero_points", 60)
    elif active_license_count == 1:
        score = license_rules.get("single_license_points", 45)
    elif active_license_count == 2:
        score = license_rules.get("two_license_points", 20)
    else:
        score = license_rules.get("three_or_more_points", 0)

    if bool_value(row.get("single_supplier_flag")):
        score += rules.get("single_supplier_points", 35)
    score += clamp(row.get("import_manufacturer_ratio"), maximum=1.0) * rules.get(
        "import_manufacturer_ratio_max_points", 15
    )
    expiring_count = 0 if pd.isna(row.get("expiring_license_count")) else row.get(
        "expiring_license_count"
    )
    score += expiring_count * rules.get("expiring_license_points_each", 10)
    return clamp(score, maximum=rules.get("max_score", 100))


def score_demand(row: pd.Series, rules: dict[str, Any]) -> float:
    if not bool_value(row.get("_demand_available")):
        return 0.0
    score = clamp(row.get("demand_rank_percentile"), maximum=1.0) * rules.get(
        "demand_rank_percentile_max_points", 70
    )
    score += clamp(row.get("demand_share_within_group"), maximum=1.0) * rules.get(
        "demand_share_within_group_max_points", 20
    )

    yoy_growth = row.get("claim_qty_yoy_growth")
    if not pd.isna(yoy_growth) and yoy_growth > 0:
        full_points_at = rules.get("yoy_growth_full_points_at", 1.0)
        score += clamp(yoy_growth / full_points_at, maximum=1.0) * rules.get(
            "yoy_growth_max_points", 10
        )

    claim_cv = row.get("claim_qty_3y_cv")
    if not pd.isna(claim_cv) and claim_cv > 0:
        full_points_at = rules.get("three_year_cv_full_points_at", 1.0)
        score += clamp(claim_cv / full_points_at, maximum=1.0) * rules.get(
            "three_year_cv_max_points", 10
        )
    return clamp(score, maximum=rules.get("max_score", 100))


def score_recall(row: pd.Series, rules: dict[str, Any]) -> float:
    if not bool_value(row.get("_recall_available")):
        return 0.0
    recall_12m = 0 if pd.isna(row.get("recall_count_12m")) else row.get("recall_count_12m")
    recall_24m = 0 if pd.isna(row.get("recall_count_24m")) else row.get("recall_count_24m")
    score = recall_12m * rules.get("recall_12m_points_each", 35)
    score += recall_24m * rules.get("recall_24m_points_each", 15)
    if bool_value(row.get("severe_recall_flag")):
        score += rules.get("severe_recall_points", 50)
    return clamp(score, maximum=rules.get("max_score", 100))


def score_optional_component(row: pd.Series, column: str) -> float:
    return clamp(row.get(column))


def available_components(row: pd.Series) -> list[str]:
    components = []
    if bool_value(row.get("_supply_available")):
        components.append("supply_score")
    if bool_value(row.get("_demand_available")):
        components.append("demand_score")
    if bool_value(row.get("_recall_available")):
        components.append("recall_score")
    if bool_value(row.get("_global_shortage_available")):
        components.append("global_shortage_score")
    if bool_value(row.get("_price_or_exit_available")):
        components.append("price_or_exit_score")
    return components


def weighted_total(row: pd.Series, weights: dict[str, float]) -> float:
    components = available_components(row)
    denominator = sum(float(weights.get(component, 0)) for component in components)
    if denominator <= 0:
        return 0.0
    total = sum(row[component] * float(weights.get(component, 0)) for component in components)
    return clamp(total / denominator)


def risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def risk_explanation(row: pd.Series) -> str:
    explanations = []
    if row["supply_score"] >= 70:
        explanations.append("high supply fragility")
    elif row["supply_score"] >= 40:
        explanations.append("moderate supply fragility")
    if row["demand_score"] >= 70:
        explanations.append("high demand pressure")
    elif row["demand_score"] >= 40:
        explanations.append("moderate demand pressure")
    if row["recall_score"] >= 70:
        explanations.append("strong recall signal")
    elif row["recall_score"] > 0:
        explanations.append("some recall signal")
    if row["global_shortage_score"] > 0:
        explanations.append("global shortage signal")
    if row["price_or_exit_score"] > 0:
        explanations.append("price or market-exit signal")
    if not explanations:
        return "Low current rule-based signal across available features."
    return "Rule-based score reflects " + ", ".join(explanations) + "."


def build_base_groups(master_path: Path) -> pd.DataFrame:
    master = pd.read_parquet(master_path)
    if "shortage_group_key" not in master.columns:
        raise ValueError("drug master must include shortage_group_key.")
    columns = ["shortage_group_key"] + [
        column for column in OPTIONAL_COMPONENT_COLUMNS if column in master.columns
    ]
    return master[columns].drop_duplicates(subset=["shortage_group_key"]).reset_index(drop=True)


def build_risk_scores(
    master_path: Path = DEFAULT_MASTER_PATH,
    supply_path: Path = DEFAULT_SUPPLY_PATH,
    demand_path: Path = DEFAULT_DEMAND_PATH,
    recall_path: Path = DEFAULT_RECALL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> ScoreRiskResult:
    config = load_config(config_path)
    weights = config["weights"]
    scores = build_base_groups(master_path)

    supply = safe_read_parquet(
        supply_path,
        [
            "shortage_group_key",
            "active_license_count",
            "manufacturer_count",
            "import_manufacturer_ratio",
            "single_supplier_flag",
            "expiring_license_count",
        ],
    )
    demand = safe_read_parquet(
        demand_path,
        [
            "shortage_group_key",
            "total_claim_qty_latest_year",
            "demand_rank_percentile",
            "demand_share_within_group",
            "claim_qty_yoy_growth",
            "claim_qty_3y_cv",
        ],
    )
    recall = safe_read_parquet(
        recall_path,
        [
            "shortage_group_key",
            "recall_count_12m",
            "recall_count_24m",
            "severe_recall_flag",
            "latest_recall_date",
            "match_confidence",
        ],
    )

    for feature_frame in (supply, demand, recall):
        scores = scores.merge(feature_frame, on="shortage_group_key", how="left")

    supply_columns = [
        "active_license_count",
        "manufacturer_count",
        "import_manufacturer_ratio",
        "single_supplier_flag",
        "expiring_license_count",
    ]
    demand_columns = [
        "total_claim_qty_latest_year",
        "demand_rank_percentile",
        "demand_share_within_group",
        "claim_qty_yoy_growth",
        "claim_qty_3y_cv",
    ]
    recall_columns = [
        "recall_count_12m",
        "recall_count_24m",
        "severe_recall_flag",
        "latest_recall_date",
        "match_confidence",
    ]
    scores["_supply_available"] = scores[supply_columns].notna().any(axis=1)
    scores["_demand_available"] = scores[demand_columns].notna().any(axis=1)
    scores["_recall_available"] = scores[recall_columns].notna().any(axis=1)
    scores["_global_shortage_available"] = scores.get("global_shortage_score", pd.Series()).notna()
    scores["_price_or_exit_available"] = scores.get("price_or_exit_score", pd.Series()).notna()
    scores["supply_score"] = scores.apply(
        lambda row: score_supply(row, config.get("supply_rules", {})), axis=1
    )
    scores["demand_score"] = scores.apply(
        lambda row: score_demand(row, config.get("demand_rules", {})), axis=1
    )
    scores["recall_score"] = scores.apply(
        lambda row: score_recall(row, config.get("recall_rules", {})), axis=1
    )
    if "global_shortage_score" not in scores.columns:
        scores["global_shortage_score"] = 0.0
    if "price_or_exit_score" not in scores.columns:
        scores["price_or_exit_score"] = 0.0
    scores["global_shortage_score"] = scores.apply(
        lambda row: score_optional_component(row, "global_shortage_score"), axis=1
    )
    scores["price_or_exit_score"] = scores.apply(
        lambda row: score_optional_component(row, "price_or_exit_score"), axis=1
    )
    scores["total_risk_score"] = scores.apply(lambda row: weighted_total(row, weights), axis=1)
    scores["risk_level"] = scores["total_risk_score"].map(risk_level)
    scores["risk_explanation"] = scores.apply(risk_explanation, axis=1)
    scores = scores[OUTPUT_COLUMNS].sort_values("shortage_group_key").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_path, index=False)

    return ScoreRiskResult(
        output_path=output_path,
        schema=OUTPUT_COLUMNS,
        row_count=len(scores),
        assumptions=[
            "Risk scoring is rule-based and configurable in configs/scoring.yml.",
            "Component scores are clamped to the 0-100 range.",
            "Total risk score is a weighted average over available components.",
            "Missing feature files or feature columns are treated as neutral zero signals.",
            "Risk levels are Low 0-39, Medium 40-69, and High 70-100.",
        ],
    )


def print_summary(result: ScoreRiskResult) -> None:
    print(f"Wrote {result.output_path}")
    print(f"Rows: {result.row_count}")
    print("Schema:")
    for column in result.schema:
        print(f"- {column}")
    print("Assumptions:")
    for assumption in result.assumptions:
        print(f"- {assumption}")


def main() -> int:
    result = build_risk_scores()
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
