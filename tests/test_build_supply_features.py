from datetime import date
from pathlib import Path

import pandas as pd

from drug_shortage.transform.build_supply_features import (
    OUTPUT_COLUMNS,
    build_supply_features,
)


def test_build_supply_features_from_drug_master_sample(tmp_path: Path) -> None:
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_supply.parquet"
    pd.DataFrame(
        [
            {
                "shortage_group_key": "A|TAB|10MG|A00",
                "license_no": "L001",
                "manufacturer": "Local A",
                "country": "TW",
            },
            {
                "shortage_group_key": "A|TAB|10MG|A00",
                "license_no": "L002",
                "manufacturer": "Import B",
                "country": "US",
            },
            {
                "shortage_group_key": "B|INJ|1ML|B00",
                "license_no": "L003",
                "manufacturer": "Only C",
                "country": "TW",
            },
        ]
    ).to_parquet(master_path, index=False)

    result = build_supply_features(
        master_path=master_path,
        output_path=output_path,
        as_of_date=date(2026, 6, 10),
    )
    features = pd.read_parquet(output_path)

    assert output_path.exists()
    assert result.schema == OUTPUT_COLUMNS
    assert features.columns.tolist() == OUTPUT_COLUMNS
    assert result.row_count == 2

    group_a = features.loc[features["shortage_group_key"] == "A|TAB|10MG|A00"].iloc[0]
    assert group_a["active_license_count"] == 2
    assert group_a["manufacturer_count"] == 2
    assert group_a["import_manufacturer_ratio"] == 0.5
    assert not group_a["single_supplier_flag"]
    assert group_a["expiring_license_count"] == 0

    group_b = features.loc[features["shortage_group_key"] == "B|INJ|1ML|B00"].iloc[0]
    assert group_b["active_license_count"] == 1
    assert group_b["manufacturer_count"] == 1
    assert group_b["single_supplier_flag"]


def test_build_supply_features_uses_expiry_dates_when_available(tmp_path: Path) -> None:
    master_path = tmp_path / "drug_master.parquet"
    output_path = tmp_path / "features_supply.parquet"
    pd.DataFrame(
        [
            {
                "shortage_group_key": "A|TAB|10MG|A00",
                "license_no": "ACTIVE_SOON",
                "manufacturer": "Local A",
                "country": "TW",
                "license_expiry_date": "2026-08-01",
            },
            {
                "shortage_group_key": "A|TAB|10MG|A00",
                "license_no": "EXPIRED",
                "manufacturer": "Local B",
                "country": "TW",
                "license_expiry_date": "2026-01-01",
            },
            {
                "shortage_group_key": "A|TAB|10MG|A00",
                "license_no": "ACTIVE_LATER",
                "manufacturer": "Import C",
                "country": "JP",
                "license_expiry_date": "2027-06-01",
            },
        ]
    ).to_parquet(master_path, index=False)

    build_supply_features(
        master_path=master_path,
        output_path=output_path,
        as_of_date=date(2026, 6, 10),
    )
    features = pd.read_parquet(output_path)

    group = features.iloc[0]
    assert group["active_license_count"] == 2
    assert group["manufacturer_count"] == 3
    assert group["expiring_license_count"] == 1
