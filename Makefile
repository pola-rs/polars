.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install --upgrade -r py-polars/requirements-dev.txt
	$(VENV_BIN)/pip install --upgrade -r py-polars/requirements-lint.txt
	$(VENV_BIN)/pip install --upgrade -r py-polars/docs/requirements-docs.txt
	$(VENV_BIN)/pip install --upgrade -r docs/requirements.txt

.PHONY: build
build: .venv  ## Compile and install Python Polars for development
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml

.PHONY: build-debug-opt
build-debug-opt: .venv  ## Compile and install Python Polars with minimal optimizations turned on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --profile opt-dev

.PHONY: build-debug-opt-subset
build-debug-opt-subset: .venv  ## Compile and install Python Polars with minimal optimizations turned on and no default features
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --no-default-features --profile opt-dev

.PHONY: build-opt
build-opt: .venv  ## Compile and install Python Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --profile debug-release

.PHONY: build-release
build-release: .venv  ## Compile and install a faster Python Polars binary with full optimizations
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --release

.PHONY: build-native
build-native: .venv  ## Same as build, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml -- -C target-cpu=native

.PHONY: build-debug-opt-native
build-debug-opt-native: .venv  ## Same as build-debug-opt, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --profile opt-dev -- -C target-cpu=native

.PHONY: build-opt-native
build-opt-native: .venv  ## Same as build-opt, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --profile debug-release -- -C target-cpu=native

.PHONY: build-release-native
build-release-native: .venv  ## Same as build-release, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop -m py-polars/Cargo.toml --release -- -C target-cpu=native

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy --all-targets --locked -- -D warnings

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	$(VENV_BIN)/ruff check .
	$(VENV_BIN)/ruff format .
	cargo fmt --all
	dprint fmt
	$(VENV_BIN)/typos .

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run all code quality checks

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf target/
	@rm -f Cargo.lock
	@cargo clean
	@$(MAKE) -s -C py-polars/ $@

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
