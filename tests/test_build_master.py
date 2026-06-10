from pathlib import Path

import pandas as pd

from drug_shortage.transform.build_master import (
    OUTPUT_COLUMNS,
    build_drug_master,
    license_key_from_text,
    license_key_from_url,
    normalize_strength,
)


def test_license_key_helpers_extract_matching_numeric_core() -> None:
    assert license_key_from_text("衛署藥製字第048352號") == "48352"
    assert license_key_from_url("https://example.test/detail?licId=01048352") == "48352"


def test_normalize_strength_compacts_common_spacing() -> None:
    assert normalize_strength("RISPERIDONE 1 MG / ML") == "RISPERIDONE 1MG/ML"


def test_build_drug_master_from_sample_data(tmp_path: Path) -> None:
    output_path = tmp_path / "drug_master.parquet"

    result = build_drug_master(output_path=output_path)
    master = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert result.row_count == 4
    assert result.join_rate == 1.0
    assert result.unmatched_records == []
    assert master.columns.tolist() == OUTPUT_COLUMNS
    assert master["drug_code"].tolist() == [
        "A030862157",
        "AC29439329",
        "AC37909329",
        "AC48352151",
    ]

    seridol = master.loc[master["drug_code"] == "AC48352151"].iloc[0]
    assert seridol["license_no"] == "衛署藥製字第048352號"
    assert seridol["ingredient_normalized"] == "RISPERIDONE"
    assert seridol["dosage_form_normalized"] == "內服液劑"
    assert seridol["strength_normalized"] == "RISPERIDONE 1MG/ML"
    assert seridol["shortage_group_key"] == "RISPERIDONE|內服液劑|RISPERIDONE 1MG/ML|N05AX08"


def test_build_drug_master_keeps_unmatched_records(tmp_path: Path) -> None:
    nhi_path = tmp_path / "nhi.csv"
    license_path = tmp_path / "licenses.csv"
    output_path = tmp_path / "drug_master.parquet"
    nhi_path.write_text(
        (
            "藥品代號,藥品中文名稱,成分,製造廠名稱,劑型,ATC代碼,有效起日,有效迄日,藥品代碼超連結\n"
            "A001,測試藥品,TEST 10 MG,測試製造,錠劑,A00AA00,1130101,9991231,\n"
        ),
        encoding="utf-8",
    )
    license_path.write_text(
        "許可證字號,中文品名,主成分略述,劑型,申請商名稱,製造商名稱,製造廠國別\n",
        encoding="utf-8",
    )

    result = build_drug_master(
        nhi_path=nhi_path,
        license_path=license_path,
        output_path=output_path,
    )
    master = pd.read_parquet(output_path)

    assert result.row_count == 1
    assert result.join_rate == 0.0
    assert result.unmatched_records == ["A001"]
    assert pd.isna(master.iloc[0]["license_no"])
    assert not master.iloc[0]["matched_license"]
