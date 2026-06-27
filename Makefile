.PHONY: install test build-bank run-web

install:
	pip install -e ".[dev]"

test:
	pytest -v

build-bank:
	python scripts/build_patch_bank.py

run-web:
	uvicorn rs_words.web:app --reload --host 0.0.0.0 --port 8000
