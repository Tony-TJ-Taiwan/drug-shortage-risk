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

The other pipeline commands are placeholders in this scaffold. They create expected working directories and confirm the command surface.

## Data Policy

Raw public CSV files live under `data/raw/` and are intentionally ignored by git. Do not delete, rename, move, overwrite, or commit those files.

Generated outputs should be written as Parquet files under `data/processed/`. Local analytical queries should use DuckDB and should not require a database server.

## Current Scope

The next implementation step is to review `docs/data_profile.md`, define stable schemas and join keys, and design explainable scoring rules before writing production transformations.
