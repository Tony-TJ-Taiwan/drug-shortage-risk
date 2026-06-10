from pathlib import Path

import pandas as pd

from drug_shortage.transform.build_demand_features import (
    OUTPUT_COLUMNS,
    build_demand_features,
)


def write_master(path: Path) -> None:
    pd.DataFrame(
        [
            {"drug_code": "A001", "shortage_group_key": "GROUP_A"},
            {"drug_code": "A002", "shortage_group_key": "GROUP_A"},
            {"drug_code": "B001", "shortage_group_key": "GROUP_B"},
        ]
    ).to_parquet(path, index=False)


def test_build_demand_features_documents_single_113_year(tmp_path: Path) -> None:
    claims_path = tmp_path / "claims.csv"
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_demand.parquet"
    write_master(master_path)
    claims_path.write_text(
        (
            "費用年,藥品代碼,含包裹支付的醫令量_合計\n"
            "113,A001,100\n"
            "113,A002,300\n"
            "113,B001,200\n"
            "113,UNMATCHED,999\n"
        ),
        encoding="utf-8",
    )

    result = build_demand_features(
        claims_path=claims_path,
        master_path=master_path,
        output_path=output_path,
    )
    features = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert result.latest_year == 113
    assert result.available_years == [113]
    assert result.row_count == 2
    assert any("Only 113-year claims data is available" in item for item in result.assumptions)
    assert features.columns.tolist() == OUTPUT_COLUMNS

    group_a = features.loc[features["shortage_group_key"] == "GROUP_A"].iloc[0]
    assert group_a["total_claim_qty_latest_year"] == 400
    assert group_a["demand_share_within_group"] == 0.75
    assert pd.isna(group_a["claim_qty_yoy_growth"])
    assert pd.isna(group_a["claim_qty_3y_cv"])

    group_b = features.loc[features["shortage_group_key"] == "GROUP_B"].iloc[0]
    assert group_b["total_claim_qty_latest_year"] == 200
    assert group_b["demand_share_within_group"] == 1.0
    assert group_b["demand_rank_percentile"] < group_a["demand_rank_percentile"]


def test_build_demand_features_calculates_multi_year_metrics(tmp_path: Path) -> None:
    claims_path = tmp_path / "claims.csv"
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_demand.parquet"
    write_master(master_path)
    claims_path.write_text(
        (
            "費用年,藥品代碼,含包裹支付的醫令量_合計\n"
            "111,A001,100\n"
            "112,A001,200\n"
            "113,A001,300\n"
            "111,B001,50\n"
            "112,B001,50\n"
            "113,B001,100\n"
        ),
        encoding="utf-8",
    )

    build_demand_features(
        claims_path=claims_path,
        master_path=master_path,
        output_path=output_path,
    )
    features = pd.read_parquet(output_path)

    group_a = features.loc[features["shortage_group_key"] == "GROUP_A"].iloc[0]
    assert group_a["total_claim_qty_latest_year"] == 300
    assert group_a["claim_qty_yoy_growth"] == 0.5
    assert round(group_a["claim_qty_3y_cv"], 6) == 0.5

    group_b = features.loc[features["shortage_group_key"] == "GROUP_B"].iloc[0]
    assert group_b["claim_qty_yoy_growth"] == 1.0
