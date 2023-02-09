.DEFAULT_GOAL := help

SHELL=/bin/bash
VENV = venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

venv:  ## Set up virtual environment
	python3 -m venv $(VENV)
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements-dev.txt
	$(VENV_BIN)/pip install -r requirements-lint.txt
	$(VENV_BIN)/pip install -r docs/requirements-docs.txt

.PHONY: build
build: venv  ## Compile and install Polars for development
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop

.PHONY: build-release
build-release: venv  ## Compile and install a faster Polars binary
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop --release

.PHONY: fmt
fmt: venv  ## Run autoformatting and linting
	$(VENV_BIN)/black .
	$(VENV_BIN)/blackdoc .
	$(VENV_BIN)/ruff .
	cargo fmt --all
	-dprint fmt
	-$(VENV_BIN)/mypy

.PHONY: clippy
clippy:  ## Run clippy
	cargo clippy -- -D warnings

.PHONY: pre-commit
pre-commit: fmt clippy  ## Run all code quality checks

.PHONY: test
test: venv build  ## Run fast unittests
	$(VENV_BIN)/pytest -n auto

.PHONY: doctest
doctest: venv build  ## Run doctests
	$(VENV_BIN)/python tests/docs/run_doctest.py

.PHONY: test-all
test-all: venv build  ## Run all tests
	$(VENV_BIN)/pytest -n auto -m "slow or not slow"
	$(VENV_BIN)/python tests/docs/run_doctest.py

.PHONY: coverage
coverage: venv build  ## Run tests and report coverage
	$(VENV_BIN)/pytest --cov -n auto -m "not benchmark"

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf venv/
	@rm -rf target/
	@rm -rf docs/build/
	@rm -rf docs/source/reference/api/
	@rm -rf .hypothesis/
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f polars/polars.abi3.so
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -delete
	@cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
