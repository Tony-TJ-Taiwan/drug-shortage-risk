from pathlib import Path

import pandas as pd
import yaml

from drug_shortage.profiling.create_sample_data import (
    REQUIRED_DATASETS,
    create_sample_data,
)


def _write_sources_config(config_path: Path) -> None:
    config_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "tfda_license": {"local_path": "data/raw/licenses.csv"},
                    "nhi_drug_items": {"local_path": "data/raw/nhi_items.csv"},
                    "nhi_claims_113": {"local_path": "data/raw/claims.csv"},
                    "recalls": {"local_path": "data/raw/recalls.csv"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_create_sample_data_preserves_columns_and_join_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    config_path = tmp_path / "sources.yml"
    sample_dir = tmp_path / "data" / "sample"
    report_path = tmp_path / "docs" / "data_profile.md"
    _write_sources_config(config_path)

    pd.DataFrame(
        [
            {"license_no": "01010398", "ingredient": "Aspirin", "dosage_form": "Tablet"},
            {"license_no": "99999999", "ingredient": "Other", "dosage_form": "Capsule"},
        ]
    ).to_csv(raw_dir / "licenses.csv", index=False)
    pd.DataFrame(
        [
            {
                "drug_code": "A001",
                "license_url": "https://example.test/detail?licId=01010398",
                "ingredient": "Aspirin",
                "dosage_form": "Tablet",
                "atc_code": "N02BA01",
            },
            {
                "drug_code": "A999",
                "license_url": "https://example.test/detail?licId=99999999",
                "ingredient": "Other",
                "dosage_form": "Capsule",
                "atc_code": "A00AA00",
            },
        ]
    ).to_csv(raw_dir / "nhi_items.csv", index=False)
    pd.DataFrame(
        [
            {"year": "113", "drug_code": "A001", "claim_qty": 10},
            {"year": "113", "drug_code": "B002", "claim_qty": 20},
        ]
    ).to_csv(raw_dir / "claims.csv", index=False)
    pd.DataFrame(
        [
            {"recall_id": "R1", "license_no": "01010398", "reason": "Quality"},
            {"recall_id": "R2", "license_no": "22222222", "reason": "Label"},
        ]
    ).to_csv(raw_dir / "recalls.csv", index=False)

    results, limitations = create_sample_data(
        config_path=config_path,
        sample_dir=sample_dir,
        report_path=report_path,
        project_root=tmp_path,
    )

    assert {result.dataset_key for result in results} == set(REQUIRED_DATASETS)
    sample_claims = pd.read_csv(sample_dir / "sample_claims.csv")
    sample_items = pd.read_csv(sample_dir / "sample_nhi_items.csv")
    sample_licenses = pd.read_csv(sample_dir / "sample_licenses.csv")
    sample_recalls = pd.read_csv(sample_dir / "sample_recalls.csv")

    assert sample_claims.columns.tolist() == ["year", "drug_code", "claim_qty"]
    assert sample_items.columns.tolist() == [
        "drug_code",
        "license_url",
        "ingredient",
        "dosage_form",
        "atc_code",
    ]
    assert "A001" in set(sample_claims["drug_code"])
    assert "A001" in set(sample_items["drug_code"])
    assert "01010398" in set(sample_licenses["license_no"].astype(str).str.zfill(8))
    assert "01010398" in set(sample_recalls["license_no"].astype(str).str.zfill(8))
    assert any("YoY growth" in limitation for limitation in limitations)
    assert "Sample Data Notes" in report_path.read_text(encoding="utf-8")


def test_create_sample_data_documents_missing_exact_joins(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    config_path = tmp_path / "sources.yml"
    _write_sources_config(config_path)

    pd.DataFrame([{"license_no": "10000000", "name": "A"}]).to_csv(
        raw_dir / "licenses.csv",
        index=False,
    )
    pd.DataFrame([{"drug_code": "A001", "license_url": "no-license-url"}]).to_csv(
        raw_dir / "nhi_items.csv",
        index=False,
    )
    pd.DataFrame([{"claim_code": "B002", "claim_qty": 1}]).to_csv(
        raw_dir / "claims.csv",
        index=False,
    )
    pd.DataFrame([{"recall_id": "R1", "license_no": "20000000"}]).to_csv(
        raw_dir / "recalls.csv",
        index=False,
    )

    _results, limitations = create_sample_data(
        config_path=config_path,
        sample_dir=tmp_path / "data" / "sample",
        report_path=tmp_path / "docs" / "data_profile.md",
        project_root=tmp_path,
    )

    assert any("No `licId` values" in limitation for limitation in limitations)
    assert any("YoY growth" in limitation for limitation in limitations)
