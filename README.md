# Drug Shortage Risk

This repository is a starter project for a public-data drug shortage risk scoring pipeline. The first version is intentionally rule-based and explainable rather than machine learning-based.

## Project Layout

```text
configs/            Source metadata and pipeline configuration
data/raw/           Local public CSV inputs, ignored by git
data/interim/       Intermediate generated files, ignored by git
data/processed/     Generated Parquet outputs, ignored by git
data/sample/        Small committed examples later
docs/               Notes and documentation
src/drug_shortage/  Python package and CLI
tests/              Pytest test suite
```

## Requirements

- Python 3.11+
- `uv`

Install dependencies:

```powershell
uv sync
```

## Commands

With `make`:

```powershell
make install
make lint
make test
make profile-data
make create-sample-data
make build-master
make build-features
make score
```

PowerShell alternatives without `make`:

```powershell
uv sync
uv run ruff check src tests
uv run pytest -q
uv run drug-shortage profile-data
uv run drug-shortage create-sample-data
uv run drug-shortage build-master
uv run drug-shortage build-features
uv run drug-shortage score
```

`profile-data` reads the CSV files listed in `configs/sources.yml`, tries common encodings, prints a concise profile, and writes `docs/data_profile.md`.

`create-sample-data` reads the local public CSV files listed in `configs/sources.yml` and writes small UTF-8 CSV fixtures under `data/sample/`:

- `data/sample/sample_licenses.csv`
- `data/sample/sample_nhi_items.csv`
- `data/sample/sample_claims.csv`
- `data/sample/sample_recalls.csv`

The command preserves source column names, keeps only a small number of rows, and prefers rows with joinable NHI item codes and license identifiers when those relationships are detectable. Any join limitations, including the current one-year claims limitation for YoY growth, are written to `docs/data_profile.md`.

`build-master` joins sample NHI drug items with sample TFDA license data, keeps the latest NHI history row per drug code, normalizes ingredient, dosage form, and strength fields, and writes `data/processed/drug_master.parquet`. Missing optional fields are left null, and missing shortage group parts are represented as `UNKNOWN` in `shortage_group_key`.

`build-features` reads `data/processed/drug_master.parquet`, calculates supply fragility features per `shortage_group_key`, and writes `data/processed/features_supply.parquet`. It also joins `data/sample/sample_claims.csv` to the master table, calculates demand pressure features, and writes `data/processed/features_demand.parquet`. Recall signal features are matched from `data/sample/sample_recalls.csv` by license number first, then by product name only when license number is missing, and written to `data/processed/features_recall.parquet`. With the current 113-year sample claims only, YoY growth and 3-year CV are documented as unavailable.

`score` reads the master and feature Parquet files, applies rule-based weights from `configs/scoring.yml`, and writes `data/processed/shortage_risk_scores.parquet`. Scores are explainable 0-100 values; the pipeline does not use machine learning.

## Data Policy

Raw public CSV files live under `data/raw/` and are intentionally ignored by git. Do not delete, rename, move, overwrite, or commit those files.

Generated outputs should be written as Parquet files under `data/processed/`. Local analytical queries should use DuckDB and should not require a database server.

## Current Scope

The next implementation step is to review `docs/data_profile.md`, define stable schemas and join keys, and design explainable scoring rules before writing production transformations.
