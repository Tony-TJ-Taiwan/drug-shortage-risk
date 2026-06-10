.PHONY: install lint test profile-data create-sample-data build-master build-features score dashboard

install:
	uv sync

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

profile-data:
	uv run drug-shortage profile-data

create-sample-data:
	uv run drug-shortage create-sample-data

build-master:
	uv run drug-shortage build-master

build-features:
	uv run drug-shortage build-features

score:
	uv run drug-shortage score

dashboard:
	uv run streamlit run src/drug_shortage/dashboard/app.py
