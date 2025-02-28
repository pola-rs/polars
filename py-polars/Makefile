.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=bash
VENV=../.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.PHONY: .venv
.venv:  ## Set up virtual environment and install requirements
	@$(MAKE) -s -C .. $@

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	@$(MAKE) -s -C .. $@

.PHONY: requirements-all
requirements-all: .venv  ## Install/refresh all Python requirements (including those needed for CI tests)
	@$(MAKE) -s -C .. $@

.PHONY: build
build: .venv  ## Compile and install Python Polars for development
	@$(MAKE) -s -C .. $@

.PHONY: build-release
build-release: .venv  ## Compile and install Python Polars binary with optimizations, with minimal debug symbols
	@$(MAKE) -s -C .. $@

.PHONY: build-nodebug-release
build-nodebug-release: .venv  ## Same as build-release, but without any debug symbols at all (a bit faster to build)
	@$(MAKE) -s -C .. $@

.PHONY: build-debug-release
build-debug-release: .venv  ## Same as build-release, but with full debug symbols turned on (a bit slower to build)
	@$(MAKE) -s -C .. $@

.PHONY: build-dist-release
build-dist-release: .venv  ## Compile and install Python Polars binary with super slow extra optimization turned on, for distribution
	@$(MAKE) -s -C .. $@
	
.PHONY: fix
fix:
	@$(MAKE) -s -C .. $@

.PHONY: lint
lint: .venv  ## Run lint checks (only)
	$(VENV_BIN)/ruff check
	-$(VENV_BIN)/mypy

.PHONY: fmt
fmt: .venv  ## Run autoformatting (and lint)
	$(VENV_BIN)/ruff check
	$(VENV_BIN)/ruff format
	$(VENV_BIN)/typos ..
	cargo fmt --all
	-dprint fmt
	-$(VENV_BIN)/mypy

.PHONY: clippy
clippy:  ## Run clippy
	cargo clippy --locked -- -D warnings -D clippy::dbg_macro

.PHONY: pre-commit
pre-commit: fmt clippy  ## Run all code formatting and lint/quality checks

.PHONY: test
test: .venv build  ## Run fast unittests
	$(VENV_BIN)/pytest -n auto --dist loadgroup $(PYTEST_ARGS)

.PHONY: test-all
test-all: .venv build  ## Run all tests
	$(VENV_BIN)/pytest -n auto --dist loadgroup -m "slow or not slow"
	$(VENV_BIN)/python tests/docs/run_doctest.py

.PHONY: doctest
doctest: .venv build  ## Run doctests
	$(VENV_BIN)/python tests/docs/run_doctest.py
	$(VENV_BIN)/pytest tests/docs/test_user_guide.py -m docs

.PHONY: docs
docs: .venv  ## Build Python docs (incremental)
	@$(MAKE) -s -C docs html

.PHONY: docs-clean
docs-clean: .venv  ## Build Python docs (full rebuild)
	@$(MAKE) -s -C docs clean
	@$(MAKE) docs

.PHONY: coverage
coverage: .venv build  ## Run tests and report coverage
	$(VENV_BIN)/pytest --cov -n auto --dist loadgroup -m "not release and not benchmark"

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@$(MAKE) -s -C docs clean
	@rm -rf .hypothesis/
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@$(VENV_BIN)/ruff clean
	@rm -rf tests/data/pdsh/sf*
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f polars/polars.abi3.so
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -exec rm -r {} +

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
