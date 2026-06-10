.PHONY: install lint test profile-data build-master build-features score

install:
	uv sync

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

profile-data:
	uv run drug-shortage profile-data

build-master:
	uv run drug-shortage build-master

build-features:
	uv run drug-shortage build-features

score:
	uv run drug-shortage score
