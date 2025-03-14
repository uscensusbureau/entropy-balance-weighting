.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV=.venv
PY_BIN=/bin/python3.11
VENV_BIN=$(VENV)/bin

.venv:
	$(PY_BIN) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install uv

.PHONY: fmt
fmt:
	$(VENV_BIN)/ruff format .
	$(VENV_BIN)/ruff check .
	-$(VENV_BIN)/mypy .

.PHONY: requirements
requirements: .venv
	$(VENV_BIN)/uv pip install -r requirements-dev.txt
	$(VENV_BIN)/uv pip install -r requirements-lint.txt

.PHONY: clean
clean:
	rm -rf .venv
