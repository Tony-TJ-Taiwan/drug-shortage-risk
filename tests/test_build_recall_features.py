from pathlib import Path

import pandas as pd

from drug_shortage.transform.build_recall_features import (
    OUTPUT_COLUMNS,
    build_recall_features,
    license_keys_from_text,
)


def write_master(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "shortage_group_key": "GROUP_A",
                "license_no": "衛署藥製字第048352號",
                "product_name": "Alpha Tablet",
            },
            {
                "shortage_group_key": "GROUP_B",
                "license_no": "衛署藥製字第030862號",
                "product_name": "Beta Syrup",
            },
            {
                "shortage_group_key": "GROUP_C",
                "license_no": None,
                "product_name": "Gamma Gel",
            },
        ]
    ).to_parquet(path, index=False)


def test_license_keys_from_text_splits_multiple_license_numbers() -> None:
    assert license_keys_from_text("衛署藥製字第006316號、衛署藥製字第023151號") == [
        "06316",
        "23151",
    ]


def test_build_recall_features_matches_license_first(tmp_path: Path) -> None:
    recalls_path = tmp_path / "recalls.csv"
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_recall.parquet"
    write_master(master_path)
    recalls_path.write_text(
        (
            "回收分級,日期,產品,許可證字號\n"
            "第一級,2026/05/01,Wrong Product Name,衛署藥製字第048352號\n"
            "第二級,2025/07/01,Beta Syrup,衛署藥製字第030862號\n"
            "第二級,2026/01/01,Gamma Gel,\n"
        ),
        encoding="utf-8",
    )

    result = build_recall_features(
        recalls_path=recalls_path,
        master_path=master_path,
        output_path=output_path,
        as_of_date=pd.Timestamp("2026-06-10"),
    )
    features = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert result.row_count == 3
    assert result.matched_recall_count == 3
    assert features.columns.tolist() == OUTPUT_COLUMNS

    group_a = features.loc[features["shortage_group_key"] == "GROUP_A"].iloc[0]
    assert group_a["recall_count_12m"] == 1
    assert group_a["recall_count_24m"] == 1
    assert group_a["severe_recall_flag"]
    assert group_a["latest_recall_date"] == pd.Timestamp("2026-05-01")
    assert group_a["match_confidence"] == "high"

    group_b = features.loc[features["shortage_group_key"] == "GROUP_B"].iloc[0]
    assert group_b["recall_count_12m"] == 1
    assert group_b["recall_count_24m"] == 1
    assert not group_b["severe_recall_flag"]
    assert group_b["match_confidence"] == "high"

    group_c = features.loc[features["shortage_group_key"] == "GROUP_C"].iloc[0]
    assert group_c["recall_count_12m"] == 1
    assert group_c["match_confidence"] == "low"


def test_build_recall_features_keeps_groups_with_no_recall(tmp_path: Path) -> None:
    recalls_path = tmp_path / "recalls.csv"
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_recall.parquet"
    write_master(master_path)
    recalls_path.write_text(
        "回收分級,日期,產品,許可證字號\n第二級,2026/05/01,No Match,衛署藥製字第999999號\n",
        encoding="utf-8",
    )

    build_recall_features(
        recalls_path=recalls_path,
        master_path=master_path,
        output_path=output_path,
        as_of_date=pd.Timestamp("2026-06-10"),
    )
    features = pd.read_parquet(output_path)

    assert features["recall_count_12m"].tolist() == [0, 0, 0]
    assert features["recall_count_24m"].tolist() == [0, 0, 0]
    assert features["match_confidence"].tolist() == ["none", "none", "none"]
