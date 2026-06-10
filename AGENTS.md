# Repository Guidelines

This repository is a Windows-friendly Python 3.11+ project for building an explainable, rule-based drug shortage risk scoring pipeline from public data.

## Ground Rules

- Do not delete, rename, move, or overwrite files in `data/raw/`.
- Do not commit raw, interim, or processed data files.
- Keep `data/sample/` available for small committed examples later.
- Prefer `pathlib.Path` over hard-coded path separators.
- Store generated outputs as Parquet files under `data/processed/`.
- Use DuckDB for local analytical SQL; do not require a database server.
- Keep the first scoring version rule-based and explainable. Do not add machine learning or deep learning dependencies.

## Useful Commands

- `uv sync`
- `uv run ruff check src tests`
- `uv run pytest -q`
- `uv run drug-shortage profile-data`
- `uv run drug-shortage build-master`
- `uv run drug-shortage build-features`
- `uv run drug-shortage score`

The `Makefile` wraps the same commands for environments where `make` is available.
