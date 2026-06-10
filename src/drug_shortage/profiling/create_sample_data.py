from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from drug_shortage.profiling.profile_data import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    PROJECT_ROOT,
    load_datasets,
    read_csv_with_encoding,
)

DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_ROW_LIMIT = 25
REQUIRED_DATASETS = {
    "tfda_license": "sample_licenses.csv",
    "nhi_drug_items": "sample_nhi_items.csv",
    "nhi_claims_113": "sample_claims.csv",
    "recalls": "sample_recalls.csv",
}
LICENSE_ID_PATTERN = re.compile(r"licid=([0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class SampleResult:
    dataset_key: str
    output_path: Path
    row_count: int
    columns: list[str]
    key_columns: list[str]


def _configured_csv_path(dataset: dict[str, Any], project_root: Path) -> Path | None:
    local_path = dataset.get("local_path")
    if not local_path:
        return None
    return project_root / local_path


def _read_configured_dataset(
    dataset_key: str,
    datasets: dict[str, dict[str, Any]],
    project_root: Path,
) -> pd.DataFrame:
    dataset = datasets.get(dataset_key)
    if not dataset:
        raise ValueError(f"Missing dataset configuration: {dataset_key}")

    csv_path = _configured_csv_path(dataset, project_root)
    if not csv_path or not csv_path.exists():
        raise FileNotFoundError(f"Configured CSV not found for {dataset_key}: {csv_path}")

    frame, _encoding = read_csv_with_encoding(csv_path)
    return frame


def _non_empty_strings(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values != ""]


def _overlap(left: pd.Series, right: pd.Series) -> int:
    left_values = set(_non_empty_strings(left))
    if not left_values:
        return 0
    right_values = set(_non_empty_strings(right))
    return len(left_values & right_values)


def _best_overlap_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_columns: list[str] | None = None,
    right_columns: list[str] | None = None,
) -> tuple[str | None, str | None, int]:
    best_left: str | None = None
    best_right: str | None = None
    best_count = 0
    for left_column in left_columns or list(left.columns):
        for right_column in right_columns or list(right.columns):
            count = _overlap(left[left_column], right[right_column])
            if count > best_count:
                best_left = str(left_column)
                best_right = str(right_column)
                best_count = count
    return best_left, best_right, best_count


def _columns_matching(frame: pd.DataFrame, *needles: str) -> list[str]:
    lowered_needles = tuple(needle.lower() for needle in needles)
    return [
        str(column)
        for column in frame.columns
        if any(needle in str(column).lower() for needle in lowered_needles)
    ]


def _columns_with_license_urls(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        samples = _non_empty_strings(frame[column]).head(100).str.lower()
        if samples.str.contains("licid=", regex=False).any():
            columns.append(str(column))
    return columns


def _license_ids_from_frame(frame: pd.DataFrame) -> set[str]:
    ids: set[str] = set()
    for column in _columns_with_license_urls(frame):
        for value in _non_empty_strings(frame[column]):
            match = LICENSE_ID_PATTERN.search(value)
            if match:
                ids.add(match.group(1))
    return ids


def _rows_containing_any(frame: pd.DataFrame, values: set[str]) -> pd.DataFrame:
    if not values:
        return frame.iloc[0:0]

    search_values = values | {value.lstrip("0") for value in values if value.lstrip("0")}
    mask = pd.Series(False, index=frame.index)
    for column in frame.columns:
        text = frame[column].astype(str)
        for value in search_values:
            mask |= text.str.contains(re.escape(value), na=False)
    return frame.loc[mask]


def _small_sample(frame: pd.DataFrame, preferred: pd.DataFrame | None = None) -> pd.DataFrame:
    if preferred is None or preferred.empty:
        sample = frame.head(SAMPLE_ROW_LIMIT)
    else:
        remaining = frame.drop(index=preferred.index, errors="ignore")
        sample = pd.concat([preferred, remaining.head(SAMPLE_ROW_LIMIT - len(preferred))])
    return sample.head(SAMPLE_ROW_LIMIT).copy()


def _write_sample(frame: pd.DataFrame, output_path: Path) -> SampleResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return SampleResult(
        dataset_key=output_path.stem.removeprefix("sample_"),
        output_path=output_path,
        row_count=len(frame),
        columns=[str(column) for column in frame.columns],
        key_columns=[],
    )


def _document_limitations(
    report_path: Path,
    limitations: list[str],
    results: list[SampleResult],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        report_path.read_text(encoding="utf-8") if report_path.exists() else "# Data Profile\n"
    )
    marker = "## Sample Data Notes"
    base = existing.split(marker, maxsplit=1)[0].rstrip()
    lines = [
        "",
        marker,
        "",
        "Generated by `uv run python -m drug_shortage.profiling.create_sample_data`.",
        "",
        "### Created Files",
        "",
    ]
    for result in results:
        lines.append(f"- `{result.output_path.as_posix()}`: {result.row_count} rows")

    lines.extend(["", "### Join Limitations", ""])
    if limitations:
        lines.extend(f"- {limitation}" for limitation in limitations)
    else:
        lines.append("- No join limitations detected while creating the sample files.")

    report_path.write_text(base + "\n".join(lines) + "\n", encoding="utf-8")


def create_sample_data(
    config_path: Path = DEFAULT_CONFIG_PATH,
    sample_dir: Path = DEFAULT_SAMPLE_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[SampleResult], list[str]]:
    datasets = load_datasets(config_path)
    frames = {
        dataset_key: _read_configured_dataset(dataset_key, datasets, project_root)
        for dataset_key in REQUIRED_DATASETS
    }

    claims = frames["nhi_claims_113"]
    nhi_items = frames["nhi_drug_items"]
    license_frame = frames["tfda_license"]
    recalls = frames["recalls"]

    limitations: list[str] = []
    claim_code_col, item_code_col, claim_item_overlap = _best_overlap_pair(claims, nhi_items)
    if not claim_code_col or not item_code_col:
        limitations.append(
            "Claims and NHI item sample rows use deterministic head rows; "
            "no exact item-code overlap was found."
        )
        selected_codes: set[str] = set()
        sample_claims = _small_sample(claims)
        sample_items = _small_sample(nhi_items)
    else:
        selected_codes = set(_non_empty_strings(claims[claim_code_col])) & set(
            _non_empty_strings(nhi_items[item_code_col])
        )
        selected_codes = set(list(selected_codes)[:SAMPLE_ROW_LIMIT])
        matching_claims = claims[claims[claim_code_col].astype(str).isin(selected_codes)]
        matching_items = nhi_items[nhi_items[item_code_col].astype(str).isin(selected_codes)]
        sample_claims = _small_sample(claims, matching_claims)
        sample_items = _small_sample(nhi_items, matching_items)
        if claim_item_overlap == 0:
            limitations.append("Claims and NHI item sample rows do not have exact item-code joins.")

    license_ids = _license_ids_from_frame(sample_items)
    license_matches = _rows_containing_any(license_frame, license_ids)
    sample_licenses = _small_sample(license_frame, license_matches)
    if license_ids and license_matches.empty:
        limitations.append(
            "NHI item license URLs expose license IDs, "
            "but no exact TFDA license row match was found."
        )
    elif not license_ids:
        limitations.append("No `licId` values were detected in the selected NHI item rows.")

    recall_license_col, license_recall_col, recall_overlap = _best_overlap_pair(
        recalls,
        license_frame,
        left_columns=_columns_matching(recalls, "license", "licence", "霅"),
        right_columns=_columns_matching(license_frame, "license", "licence", "霅"),
    )
    if recall_license_col and license_recall_col and recall_overlap > 0:
        recall_values = set(_non_empty_strings(sample_licenses[license_recall_col]))
        sample_recalls = _small_sample(
            recalls,
            recalls[recalls[recall_license_col].astype(str).isin(recall_values)],
        )
    else:
        sample_recalls = _small_sample(recalls)
        limitations.append(
            "Recall rows could not be joined exactly to TFDA license rows "
            "with detected license columns."
        )

    if len([key for key in REQUIRED_DATASETS if key.startswith("nhi_claims_")]) == 1:
        limitations.append(
            "Only one claims year is configured, so YoY growth cannot be calculated yet."
        )

    output_frames = {
        "tfda_license": sample_licenses,
        "nhi_drug_items": sample_items,
        "nhi_claims_113": sample_claims,
        "recalls": sample_recalls,
    }
    results: list[SampleResult] = []
    for dataset_key, output_name in REQUIRED_DATASETS.items():
        result = _write_sample(output_frames[dataset_key], sample_dir / output_name)
        key_columns: list[str] = []
        if dataset_key == "nhi_claims_113" and claim_code_col:
            key_columns.append(claim_code_col)
        if dataset_key == "nhi_drug_items" and item_code_col:
            key_columns.append(item_code_col)
            key_columns.extend(_columns_with_license_urls(output_frames[dataset_key]))
        results.append(
            SampleResult(
                dataset_key=dataset_key,
                output_path=result.output_path,
                row_count=result.row_count,
                columns=result.columns,
                key_columns=key_columns,
            )
        )

    _document_limitations(report_path, limitations, results)
    return results, limitations


def print_summary(results: list[SampleResult], limitations: list[str]) -> None:
    print("Sample files created:")
    for result in results:
        key_columns = ", ".join(result.key_columns) or "-"
        print(
            f"- {result.output_path}: rows={result.row_count}, "
            f"columns={len(result.columns)}, key_columns={key_columns}"
        )

    if limitations:
        print("Join limitations:")
        for limitation in limitations:
            print(f"- {limitation}")
    else:
        print("Join limitations: none detected")


def main() -> int:
    results, limitations = create_sample_data()
    print_summary(results, limitations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
