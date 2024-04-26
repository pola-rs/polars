.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Define command to filter pip warnings when running maturin
FILTER_PIP_WARNINGS=| grep -v "don't match your environment"; test $${PIPESTATUS[0]} -eq 0

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/python -m pip install --upgrade uv \
	&& $(VENV_BIN)/uv pip install --upgrade --compile-bytecode -r py-polars/requirements-dev.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r py-polars/requirements-lint.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r py-polars/docs/requirements-docs.txt \
	&& $(VENV_BIN)/uv pip install --upgrade -r docs/requirements.txt

.PHONY: build
build: .venv  ## Compile and install Python Polars for development
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-debug-opt
build-debug-opt: .venv  ## Compile and install Python Polars with minimal optimizations turned on
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile opt-dev \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-debug-opt-subset
build-debug-opt-subset: .venv  ## Compile and install Python Polars with minimal optimizations turned on and no default features
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --no-default-features --profile opt-dev \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-opt
build-opt: .venv  ## Compile and install Python Polars with nearly full optimization on and debug assertions turned off, but with debug symbols on
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile debug-release \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-release
build-release: .venv  ## Compile and install a faster Python Polars binary with full optimizations
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --release \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-native
build-native: .venv  ## Same as build, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && RUSTFLAGS='-C target-cpu=native' \
	$(VENV_BIN)/maturin develop -m py-polars/Cargo.toml \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-debug-opt-native
build-debug-opt-native: .venv  ## Same as build-debug-opt, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && RUSTFLAGS='-C target-cpu=native' \
	$(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile opt-dev \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-opt-native
build-opt-native: .venv  ## Same as build-opt, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && RUSTFLAGS='-C target-cpu=native' \
	$(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile debug-release \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-release-native
build-release-native: .venv  ## Same as build-release, except with native CPU optimizations turned on
	@unset CONDA_PREFIX && RUSTFLAGS='-C target-cpu=native' \
	$(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --release \
	$(FILTER_PIP_WARNINGS)


.PHONY: check
check:  ## Run cargo check with all features
	cargo check --workspace --all-targets --all-features

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings -D clippy::dbg_macro

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy --all-targets --locked -- -D warnings -D clippy::dbg_macro

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	$(VENV_BIN)/ruff check
	$(VENV_BIN)/ruff format
	cargo fmt --all
	dprint fmt
	$(VENV_BIN)/typos

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run all code quality checks

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@$(MAKE) -s -C py-polars/ $@
	@rm -rf .ruff_cache/
	@rm -rf .venv/
	@cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
