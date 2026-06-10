from pathlib import Path

import pandas as pd

from drug_shortage.transform.build_master import (
    DEFAULT_LICENSE_PATH,
    DEFAULT_NHI_PATH,
    OUTPUT_COLUMNS,
    build_drug_master,
    license_key_from_text,
    license_key_from_url,
    normalize_strength,
    read_csv,
)


def test_license_key_helpers_extract_matching_numeric_core() -> None:
    assert license_key_from_text("license 01048352") == "48352"
    assert license_key_from_url("https://example.test/detail?licId=01048352") == "48352"


def test_normalize_strength_compacts_common_spacing() -> None:
    assert normalize_strength("RISPERIDONE 1 MG / ML") == "RISPERIDONE 1MG/ML"


def test_build_drug_master_from_sample_data(tmp_path: Path) -> None:
    output_path = tmp_path / "drug_master.parquet"

    result = build_drug_master(output_path=output_path)
    master = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert result.row_count == len(master)
    assert 0 <= result.join_rate <= 1
    assert isinstance(result.unmatched_records, list)
    assert master.columns.tolist() == OUTPUT_COLUMNS
    assert "drug_code" in master.columns
    assert master["drug_code"].notna().all()
    assert "shortage_group_key" in master.columns
    assert master["shortage_group_key"].notna().all()
    assert "matched_license" in master.columns


def test_build_drug_master_keeps_unmatched_records(tmp_path: Path) -> None:
    nhi_path = tmp_path / "nhi.csv"
    license_path = tmp_path / "licenses.csv"
    output_path = tmp_path / "drug_master.parquet"
    nhi_columns = read_csv(DEFAULT_NHI_PATH).columns.tolist()
    license_columns = read_csv(DEFAULT_LICENSE_PATH).columns.tolist()

    nhi_row = dict.fromkeys(nhi_columns, "")
    nhi_row[nhi_columns[1]] = "A001"
    nhi_row[nhi_columns[2]] = "TEST DRUG"
    nhi_row[nhi_columns[4]] = "TEST 10 MG"
    nhi_row[nhi_columns[13]] = "TABLET"
    nhi_row[nhi_columns[16]] = "A00AA00"
    nhi_row[nhi_columns[18]] = "https://example.test/detail?licId=9991231"
    pd.DataFrame([nhi_row], columns=nhi_columns).to_csv(
        nhi_path,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(columns=license_columns).to_csv(
        license_path,
        index=False,
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
