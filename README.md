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
uv run drug-shortage build-master
uv run drug-shortage build-features
uv run drug-shortage score
```

The pipeline commands are placeholders in this scaffold. They create expected working directories and confirm the command surface without reading or processing `data/raw/`.

## Data Policy

Raw public CSV files live under `data/raw/` and are intentionally ignored by git. Do not delete, rename, move, overwrite, or commit those files.

Generated outputs should be written as Parquet files under `data/processed/`. Local analytical queries should use DuckDB and should not require a database server.

## Current Scope

This scaffold prepares the repository structure only. The next implementation step is to profile the raw public CSV files, define stable schemas, and design explainable scoring rules before writing any production transformations.
