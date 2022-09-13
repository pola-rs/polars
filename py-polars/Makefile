.DEFAULT_GOAL := help

SHELL=/bin/bash

.PHONY: build
build:  # Build and install Polars
	maturin develop

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	isort .
	black .
	blackdoc .
	pyupgrade --py37-plus `find polars/ tests/ -name "*.py" -type f`
	cargo fmt --all
	-dprint fmt
	-mypy
	-flake8 polars tests docs

.PHONY: clippy
clippy:  ## Lint Rust files
	cargo clippy -- -D warnings -A clippy::borrow_deref_ref

.PHONY: pre-commit
pre-commit: fmt clippy ## Run all code quality checks

.PHONY: test
test: build  ## Run fast unittests
	pytest tests/unit/

.PHONY: doctest
doctest: build  ## Run doctests
	python tests/docs/run_doc_examples.py

.PHONY: test-all
test-all: build  ## Run all tests
	pytest
	python tests/docs/run_doc_examples.py

.PHONY: coverage
coverage: build  ## Run tests and report coverage
	pytest --cov

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf *venv/
	@rm -rf target/
	@rm -rf docs/build/
	@rm -rf docs/source/reference/api/
	@rm -rf .hypothesis/
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@rm -f .coverage
	@rm -f coverage.xml
	@rm -f polars/polars.abi3.so
	@find -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -delete
	@cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e '\033[1mAvailable commands:\033[0m'
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
