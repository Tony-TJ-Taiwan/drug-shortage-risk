from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sources.yml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "data_profile.md"
COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5")
SAMPLE_VALUE_LIMIT = 5
JOIN_KEYWORDS = (
    "code",
    "id",
    "license",
    "licence",
    "permit",
    "number",
    "no",
    "atc",
    "url",
    "代碼",
    "代号",
    "代號",
    "字號",
    "字号",
    "文號",
    "文号",
    "許可證",
    "许可证",
    "連結",
    "链接",
)


def load_datasets(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("configs/sources.yml must contain a 'datasets' mapping.")
    return datasets


def read_csv_with_encoding(csv_path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in COMMON_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding, low_memory=False), encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error.reason}")

    error_text = "; ".join(errors)
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode {csv_path} with supported encodings. {error_text}",
    )


def sample_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).drop_duplicates().head(SAMPLE_VALUE_LIMIT)
    return values.tolist()


def candidate_join_columns(
    columns: list[str],
    column_samples: dict[str, list[str]] | None = None,
) -> list[str]:
    candidates: list[str] = []
    samples_by_column = column_samples or {}

    for column in columns:
        normalized = column.lower().strip()
        samples = samples_by_column.get(column, [])
        sample_text = " ".join(samples).lower()

        keyword_match = any(keyword in normalized for keyword in JOIN_KEYWORDS)
        url_identifier_match = "licid=" in sample_text or "drugfilename=" in sample_text
        if keyword_match or url_identifier_match:
            candidates.append(column)

    return candidates


def profile_frame(
    dataset_key: str,
    dataset: dict[str, Any],
    frame: pd.DataFrame,
    encoding: str,
) -> dict[str, Any]:
    column_profiles = []
    row_count = len(frame)
    column_samples: dict[str, list[str]] = {}

    for column in frame.columns:
        column_name = str(column)
        samples = sample_values(frame[column])
        column_samples[column_name] = samples
        missing_count = int(frame[column].isna().sum())
        missing_ratio = float(missing_count / row_count) if row_count else 0.0
        column_profiles.append(
            {
                "name": column_name,
                "missing_ratio": missing_ratio,
                "sample_values": samples,
            }
        )

    columns = [str(column) for column in frame.columns]
    return {
        "key": dataset_key,
        "name": dataset.get("name", dataset_key),
        "local_path": dataset.get("local_path"),
        "status": "profiled",
        "encoding": encoding,
        "row_count": row_count,
        "column_count": len(frame.columns),
        "columns": columns,
        "column_profiles": column_profiles,
        "candidate_join_columns": candidate_join_columns(columns, column_samples),
    }


def profile_dataset(
    dataset_key: str,
    dataset: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    local_path = dataset.get("local_path")
    if not local_path:
        return {
            "key": dataset_key,
            "name": dataset.get("name", dataset_key),
            "local_path": local_path,
            "status": "skipped",
            "message": "No local_path configured.",
        }

    csv_path = project_root / local_path
    if not csv_path.exists():
        return {
            "key": dataset_key,
            "name": dataset.get("name", dataset_key),
            "local_path": local_path,
            "status": "missing",
            "message": f"File not found: {csv_path}",
        }

    try:
        frame, encoding = read_csv_with_encoding(csv_path)
    except UnicodeDecodeError as error:
        return {
            "key": dataset_key,
            "name": dataset.get("name", dataset_key),
            "local_path": local_path,
            "status": "error",
            "message": str(error),
        }

    return profile_frame(dataset_key, dataset, frame, encoding)


def _markdown_cell(value: Any) -> str:
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")
    return text


def render_markdown(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# Data Profile",
        "",
        "Generated by `uv run python -m drug_shortage.profiling.profile_data`.",
        "",
        "## Summary",
        "",
        "| Dataset | Status | Rows | Columns | Encoding | Candidate Join Columns |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]

    for profile in profiles:
        candidate_columns = ", ".join(profile.get("candidate_join_columns", [])) or "-"
        lines.append(
            "| {key} | {status} | {rows} | {columns} | {encoding} | {candidates} |".format(
                key=_markdown_cell(profile["key"]),
                status=_markdown_cell(profile["status"]),
                rows=profile.get("row_count", "-"),
                columns=profile.get("column_count", "-"),
                encoding=_markdown_cell(profile.get("encoding", "-")),
                candidates=_markdown_cell(candidate_columns),
            )
        )

    for profile in profiles:
        lines.extend(
            [
                "",
                f"## {_markdown_cell(profile['key'])}",
                "",
                f"- Name: {_markdown_cell(profile.get('name', profile['key']))}",
                f"- Local path: `{_markdown_cell(profile.get('local_path'))}`",
                f"- Status: {_markdown_cell(profile['status'])}",
            ]
        )

        if profile["status"] != "profiled":
            lines.append(f"- Note: {_markdown_cell(profile.get('message', 'Not profiled.'))}")
            continue

        candidate_columns = ", ".join(profile.get("candidate_join_columns", [])) or "-"
        lines.extend(
            [
                f"- Detected encoding: `{_markdown_cell(profile['encoding'])}`",
                f"- Rows: {profile['row_count']}",
                f"- Columns: {profile['column_count']}",
                f"- Candidate join columns: {_markdown_cell(candidate_columns)}",
                "",
                "### Columns",
                "",
            ]
        )
        for column in profile["columns"]:
            lines.append(f"- `{_markdown_cell(column)}`")

        lines.extend(
            [
                "",
                "### Missing Values and Samples",
                "",
                "| Column | Missing Ratio | Sample Values |",
                "| --- | ---: | --- |",
            ]
        )
        for column_profile in profile["column_profiles"]:
            samples = ", ".join(
                f"`{_markdown_cell(value)}`" for value in column_profile["sample_values"]
            )
            missing_ratio = column_profile["missing_ratio"]
            lines.append(
                "| `{name}` | {missing_ratio:.2%} | {samples} |".format(
                    name=_markdown_cell(column_profile["name"]),
                    missing_ratio=missing_ratio,
                    samples=samples or "-",
                )
            )

    return "\n".join(lines) + "\n"


def profile_all(
    config_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    datasets = load_datasets(config_path)
    profiles = [
        profile_dataset(dataset_key, dataset, project_root)
        for dataset_key, dataset in datasets.items()
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(profiles), encoding="utf-8")
    return profiles


def print_summary(
    profiles: list[dict[str, Any]],
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    print(f"Profiling report written to: {report_path}")
    for profile in profiles:
        if profile["status"] == "profiled":
            candidates = ", ".join(profile.get("candidate_join_columns", [])) or "-"
            print(
                f"{profile['key']}: rows={profile['row_count']}, "
                f"columns={profile['column_count']}, encoding={profile['encoding']}, "
                f"candidate_join_columns={candidates}"
            )
        else:
            message = profile.get("message", "Not profiled.")
            print(f"{profile['key']}: {profile['status']} - {message}")


def main() -> int:
    profiles = profile_all()
    print_summary(profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
