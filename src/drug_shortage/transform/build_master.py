from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NHI_PATH = PROJECT_ROOT / "data" / "sample" / "sample_nhi_items.csv"
DEFAULT_LICENSE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_licenses.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "drug_master.parquet"

OUTPUT_COLUMNS = [
    "drug_code",
    "license_no",
    "product_name",
    "ingredient",
    "dosage_form",
    "strength",
    "atc_code",
    "manufacturer",
    "applicant",
    "country",
    "ingredient_normalized",
    "dosage_form_normalized",
    "strength_normalized",
    "shortage_group_key",
    "matched_license",
]


@dataclass(frozen=True)
class BuildMasterResult:
    output_path: Path
    schema: list[str]
    row_count: int
    join_rate: float
    unmatched_records: list[str]
    assumptions: list[str]


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("；", ";").replace("，", ",")
    return re.sub(r"\s+", " ", text).upper()


def normalize_strength(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace("．", ".")
    text = re.sub(r"(?<=\d)\s+(?=[A-Z%])", "", text)
    text = re.sub(r"(?<=[A-Z%])\s*/\s*(?=[A-Z])", "/", text)
    text = re.sub(r"\b0+(\d)", r"\1", text)
    return text


def license_key_from_text(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = "".join(re.findall(r"\d+", text))
    if len(digits) >= 5:
        return digits[-5:]
    return None


def license_key_from_url(value: object) -> str | None:
    text = str(value).strip() if not pd.isna(value) else ""
    if not text:
        return None
    query = parse_qs(urlparse(text).query)
    lic_id = query.get("licId", [None])[0]
    return license_key_from_text(lic_id)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def nullable_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def first_available(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    return nullable_column(frame, primary).combine_first(nullable_column(frame, fallback))


def pick_latest_nhi_rows(nhi_items: pd.DataFrame) -> pd.DataFrame:
    frame = nhi_items.copy()
    frame["_effective_end_sort"] = pd.to_numeric(frame.get("有效迄日"), errors="coerce")
    frame["_effective_start_sort"] = pd.to_numeric(frame.get("有效起日"), errors="coerce")
    frame = frame.sort_values(
        ["藥品代號", "_effective_end_sort", "_effective_start_sort"],
        ascending=[True, False, False],
        na_position="last",
    )
    return frame.drop_duplicates(subset=["藥品代號"], keep="first")


def prepare_licenses(licenses: pd.DataFrame) -> pd.DataFrame:
    frame = licenses.copy()
    frame["_license_key"] = frame.get("許可證字號", pd.Series(dtype=str)).map(license_key_from_text)
    return frame.drop_duplicates(subset=["_license_key"], keep="first")


def build_shortage_group_key(row: pd.Series) -> str:
    parts = [
        row.get("ingredient_normalized"),
        row.get("dosage_form_normalized"),
        row.get("strength_normalized"),
        row.get("atc_code"),
    ]
    return "|".join(part or "UNKNOWN" for part in parts)


def build_drug_master(
    nhi_path: Path = DEFAULT_NHI_PATH,
    license_path: Path = DEFAULT_LICENSE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> BuildMasterResult:
    nhi_items = pick_latest_nhi_rows(read_csv(nhi_path))
    licenses = prepare_licenses(read_csv(license_path))

    nhi_items = nhi_items.copy()
    nhi_items["_license_key"] = nhi_items.get("藥品代碼超連結", pd.Series(dtype=str)).map(
        license_key_from_url
    )
    joined = nhi_items.merge(licenses, on="_license_key", how="left", suffixes=("_nhi", "_license"))

    master = pd.DataFrame(
        {
            "drug_code": nullable_column(joined, "藥品代號"),
            "license_no": nullable_column(joined, "許可證字號"),
            "product_name": first_available(joined, "藥品中文名稱", "中文品名"),
            "ingredient": first_available(joined, "主成分略述", "成分"),
            "dosage_form": first_available(joined, "劑型_nhi", "劑型_license"),
            "strength": nullable_column(joined, "成分"),
            "atc_code": nullable_column(joined, "ATC代碼"),
            "manufacturer": first_available(joined, "製造商名稱", "製造廠名稱"),
            "applicant": first_available(joined, "申請商名稱", "藥商"),
            "country": nullable_column(joined, "製造廠國別"),
        }
    )
    master["ingredient_normalized"] = master["ingredient"].map(normalize_text)
    master["dosage_form_normalized"] = master["dosage_form"].map(normalize_text)
    master["strength_normalized"] = master["strength"].map(normalize_strength)
    master["atc_code"] = master["atc_code"].map(normalize_text)
    master["shortage_group_key"] = master.apply(build_shortage_group_key, axis=1)
    master["matched_license"] = master["license_no"].notna()
    master = master[OUTPUT_COLUMNS].sort_values("drug_code").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(output_path, index=False)

    matched_count = int(master["matched_license"].sum())
    row_count = len(master)
    return BuildMasterResult(
        output_path=output_path,
        schema=OUTPUT_COLUMNS,
        row_count=row_count,
        join_rate=matched_count / row_count if row_count else 0.0,
        unmatched_records=master.loc[~master["matched_license"], "drug_code"].dropna().tolist(),
        assumptions=[
            (
                "NHI item history is collapsed to the latest row per drug_code using "
                "effective end date."
            ),
            (
                "License matching uses licId from the NHI hyperlink and the numeric "
                "core of TFDA license_no."
            ),
            (
                "When NHI and license fields overlap, TFDA license data is preferred "
                "for ingredient, manufacturer, applicant, and country."
            ),
            (
                "Missing fields remain null; normalized group keys use UNKNOWN for "
                "missing normalized parts."
            ),
        ],
    )


def print_summary(result: BuildMasterResult) -> None:
    print(f"Wrote {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Join rate: {result.join_rate:.2%}")
    print(f"Unmatched records: {len(result.unmatched_records)}")
    if result.unmatched_records:
        print(", ".join(result.unmatched_records))
    print("Schema:")
    for column in result.schema:
        print(f"- {column}")
    print("Assumptions:")
    for assumption in result.assumptions:
        print(f"- {assumption}")


def main() -> int:
    result = build_drug_master()
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
