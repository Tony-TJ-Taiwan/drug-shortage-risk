from pathlib import Path

import pandas as pd

from drug_shortage.transform.score_risk import OUTPUT_COLUMNS, build_risk_scores, risk_level


def test_risk_level_thresholds() -> None:
    assert risk_level(0) == "Low"
    assert risk_level(39) == "Low"
    assert risk_level(40) == "Medium"
    assert risk_level(69) == "Medium"
    assert risk_level(70) == "High"
    assert risk_level(100) == "High"


def test_build_risk_scores_combines_rule_based_components(tmp_path: Path) -> None:
    master_path = tmp_path / "drug_master.parquet"
    supply_path = tmp_path / "features_supply.parquet"
    demand_path = tmp_path / "features_demand.parquet"
    recall_path = tmp_path / "features_recall.parquet"
    output_path = tmp_path / "shortage_risk_scores.parquet"
    pd.DataFrame([{"shortage_group_key": "GROUP_A"}]).to_parquet(master_path, index=False)
    pd.DataFrame(
        [
            {
                "shortage_group_key": "GROUP_A",
                "active_license_count": 1,
                "manufacturer_count": 1,
                "import_manufacturer_ratio": 0.0,
                "single_supplier_flag": True,
                "expiring_license_count": 0,
            }
        ]
    ).to_parquet(supply_path, index=False)
    pd.DataFrame(
        [
            {
                "shortage_group_key": "GROUP_A",
                "total_claim_qty_latest_year": 1000,
                "demand_rank_percentile": 1.0,
                "demand_share_within_group": 1.0,
                "claim_qty_yoy_growth": pd.NA,
                "claim_qty_3y_cv": pd.NA,
            }
        ]
    ).to_parquet(demand_path, index=False)
    pd.DataFrame(
        [
            {
                "shortage_group_key": "GROUP_A",
                "recall_count_12m": 1,
                "recall_count_24m": 1,
                "severe_recall_flag": True,
                "latest_recall_date": "2026-05-01",
                "match_confidence": "high",
            }
        ]
    ).to_parquet(recall_path, index=False)

    result = build_risk_scores(
        master_path=master_path,
        supply_path=supply_path,
        demand_path=demand_path,
        recall_path=recall_path,
        output_path=output_path,
    )
    scores = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert scores.columns.tolist() == OUTPUT_COLUMNS
    assert result.row_count == 1
    row = scores.iloc[0]
    assert row["supply_score"] == 80
    assert row["demand_score"] == 90
    assert row["recall_score"] == 100
    assert 0 <= row["total_risk_score"] <= 100
    assert row["risk_level"] == "High"
    assert "rule-based" in row["risk_explanation"].lower()


def test_build_risk_scores_handles_missing_feature_files(tmp_path: Path) -> None:
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "shortage_risk_scores.parquet"
    pd.DataFrame([{"shortage_group_key": "GROUP_A"}]).to_parquet(master_path, index=False)

    build_risk_scores(
        master_path=master_path,
        supply_path=tmp_path / "missing_supply.parquet",
        demand_path=tmp_path / "missing_demand.parquet",
        recall_path=tmp_path / "missing_recall.parquet",
        output_path=output_path,
    )
    scores = pd.read_parquet(output_path)

    row = scores.iloc[0]
    assert row["supply_score"] == 0
    assert row["demand_score"] == 0
    assert row["recall_score"] == 0
    assert row["total_risk_score"] == 0
    assert row["risk_level"] == "Low"


def test_build_risk_scores_uses_global_shortage_score_when_available(tmp_path: Path) -> None:
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "shortage_risk_scores.parquet"
    pd.DataFrame(
        [{"shortage_group_key": "GROUP_A", "global_shortage_score": 100}]
    ).to_parquet(master_path, index=False)

    build_risk_scores(
        master_path=master_path,
        supply_path=tmp_path / "missing_supply.parquet",
        demand_path=tmp_path / "missing_demand.parquet",
        recall_path=tmp_path / "missing_recall.parquet",
        output_path=output_path,
    )
    scores = pd.read_parquet(output_path)

    row = scores.iloc[0]
    assert row["global_shortage_score"] == 100
    assert row["total_risk_score"] == 100
    assert row["risk_level"] == "High"
