from pathlib import Path

import yaml

from drug_shortage.profiling.profile_data import (
    load_datasets,
    profile_all,
    profile_dataset,
    read_csv_with_encoding,
)


def test_read_csv_with_encoding_supports_cp950(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_bytes("藥品代碼,名稱\nA001,測試藥\n".encode("cp950"))

    frame, encoding = read_csv_with_encoding(csv_path)

    assert encoding == "cp950"
    assert frame.columns.tolist() == ["藥品代碼", "名稱"]
    assert frame.iloc[0]["名稱"] == "測試藥"


def test_profile_dataset_handles_missing_file(tmp_path: Path) -> None:
    profile = profile_dataset(
        "missing_dataset",
        {"name": "Missing", "local_path": "data/raw/missing.csv"},
        project_root=tmp_path,
    )

    assert profile["status"] == "missing"
    assert "File not found" in profile["message"]


def test_profile_all_writes_markdown_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "sample.csv"
    csv_path.write_text("drug_code,name,quantity\nA001,Aspirin,10\nA002,,5\n", encoding="utf-8")

    config_path = tmp_path / "sources.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "sample": {
                        "name": "Sample",
                        "local_path": "data/raw/sample.csv",
                        "source_type": "public_csv",
                    }
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "docs" / "data_profile.md"

    profiles = profile_all(config_path=config_path, report_path=report_path, project_root=tmp_path)

    assert profiles[0]["status"] == "profiled"
    assert profiles[0]["row_count"] == 2
    assert profiles[0]["column_count"] == 3
    assert profiles[0]["candidate_join_columns"] == ["drug_code"]
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Data Profile" in report_text
    assert "`drug_code`" in report_text


def test_load_datasets_requires_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.yml"
    config_path.write_text("datasets: []\n", encoding="utf-8")

    try:
        load_datasets(config_path)
    except ValueError as error:
        assert "datasets" in str(error)
    else:
        raise AssertionError("Expected ValueError for non-mapping datasets config.")
